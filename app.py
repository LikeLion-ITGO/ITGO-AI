# app.py  (Flask, ONNX + OCR 통합 서버)
import os

os.environ["PADDLE_LOG_LEVEL"] = "3"  # 0=DEBUG, 3=ERROR

from typing import Any, Dict, List, Tuple, Optional
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import onnxruntime as ort
from time import perf_counter

# === OCR 관련 (지연 로딩) ===
try:
    import cv2
    from paddleocr import PaddleOCR
except Exception:
    cv2 = None
    PaddleOCR = None

# =========================
# Config
# =========================
MODEL_PATH = os.getenv(
    "MODEL_PATH", os.path.join("model", "meat_fresh_classifier.onnx")
)
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))
MAX_FILES = int(os.getenv("MAX_FILES", "10"))

USE_TRT = os.getenv("USE_TRT", "0") == "1"
USE_CUDA = os.getenv("USE_CUDA", "1") == "1"
PREPROCESSOR = os.getenv("PREPROCESSOR", "none").lower()

CLASS_NAMES = ["Fresh", "Half-Fresh", "Spoiled"]
TIE_PRIORITY = ["Spoiled", "Half-Fresh", "Fresh"]

app = Flask(__name__)
app.url_map.strict_slashes = False


# =========================
# Engine (ONNX only)
# =========================
def _detect_layout(input_shape: List[Any]) -> str:
    if input_shape and input_shape[-1] == 3:
        return "NHWC"
    if len(input_shape) >= 2 and input_shape[1] == 3:
        return "NCHW"
    return "NHWC"


def init_engine(path: str) -> Dict[str, Any]:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    available = ort.get_available_providers()
    providers: List[Any] = []

    if USE_TRT and ("TensorrtExecutionProvider" in available):
        trt_opts = {
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": os.getenv("TRT_CACHE", "./trt_cache"),
            "trt_max_workspace_size": int(os.getenv("TRT_WORKSPACE", "2147483648")),
        }
        os.makedirs(trt_opts["trt_engine_cache_path"], exist_ok=True)
        providers.append(("TensorrtExecutionProvider", trt_opts))

    if USE_CUDA and ("CUDAExecutionProvider" in available):
        providers.append(
            ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"})
        )

    providers.append("CPUExecutionProvider")

    sess = ort.InferenceSession(path, sess_options=so, providers=providers)
    using = sess.get_providers()
    inp_meta = sess.get_inputs()[0]
    inp_name = inp_meta.name
    inp_shape = list(inp_meta.shape)
    layout = _detect_layout(inp_shape)

    print(
        f"[ENGINE] ONNX available={available}, using={using}, input_shape={inp_shape}, layout={layout}"
    )

    out_name = sess.get_outputs()[0].name
    return {
        "type": "onnx",
        "sess": sess,
        "inp": inp_name,
        "out": out_name,
        "available_providers": available,
        "using_providers": using,
        "layout": layout,
        "use_trt": USE_TRT,
        "use_cuda": USE_CUDA,
        "input_shape": inp_shape,
    }


ENGINE: Dict[str, Any] = init_engine(MODEL_PATH)

if os.getenv("WARMUP", "0") == "1":
    dummy = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    if ENGINE["layout"] == "NCHW":
        dummy = np.transpose(dummy, (0, 3, 1, 2))
    ENGINE["sess"].run([ENGINE["out"]], {ENGINE["inp"]: dummy})


# =========================
# Meet-fresh Preprocess/Inference
# =========================
def _read_image_to_numpy(file_storage) -> np.ndarray:
    img = Image.open(file_storage.stream).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    return arr


def preprocess_batch(batch_np: np.ndarray, mode: str) -> np.ndarray:
    mode = (mode or "none").lower()
    if mode == "none":
        x = batch_np.astype(np.float32)  # 0..255
    elif mode == "raw01":
        x = batch_np.astype(np.float32) / 255.0  # 0..1
    else:  # efficientnet
        x = batch_np.astype(np.float32) / 255.0
        x = (x - 0.5) * 2.0  # -1..1
    return x


def _predict_meet_fresh(batch_np: np.ndarray) -> np.ndarray:
    if ENGINE["layout"] == "NCHW" and batch_np.ndim == 4:
        batch_np = np.transpose(batch_np, (0, 3, 1, 2))
    logits_or_probs = ENGINE["sess"].run([ENGINE["out"]], {ENGINE["inp"]: batch_np})[0]
    e = np.exp(logits_or_probs - np.max(logits_or_probs, axis=-1, keepdims=True))
    probs = e / np.sum(e, axis=-1, keepdims=True)
    return probs


def _label_from_row(row: np.ndarray) -> str:
    return CLASS_NAMES[int(np.argmax(row))]


def _aggregate_with_spoiled_rule(labels: List[str]) -> str:
    if any(l == "Spoiled" for l in labels):
        return "Spoiled"
    counts = {k: 0 for k in CLASS_NAMES}
    for l in labels:
        counts[l] += 1
    maxc = max(counts.values())
    cands = [k for k, v in counts.items() if v == maxc]
    for pri in TIE_PRIORITY:
        if pri in cands:
            return pri
    return "Spoiled"


# =========================
# OCR + KIE
# =========================
OCR_ENGINE = None  # 지연 초기화 (GPU 우선, 실패 시 CPU)

PANEL_KEYWORDS = [
    "원재료",
    "영양",
    "보관",
    "보관방법",
    "식품유형",
    "식품의 유형",
    "내용량",
    "제조원",
    "판매원",
    "수입원",
    "알레르기",
    "유통기한",
    "포장재질",
    "원산지",
]
BLOCKLIST_LABELS = [
    "원재료",
    "영양",
    "보관",
    "주의사항",
    "알레르기",
    "포장재질",
    "내용량",
    "식품유형",
    "식품의 유형",
]

import re, unicodedata

# === URL/연락처/도메인 필터 ===
URL_PAT = re.compile(r"(https?://|www\.)|([A-Za-z0-9-]+\.[A-Za-z]{2,})", re.I)
EMAIL_PAT = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", re.I)
PHONE_PAT = re.compile(r"\b\d{2,4}-\d{3,4}-\d{4}\b")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _looks_like_contact_or_url(s: str) -> bool:
    t = _norm(s)
    return bool(URL_PAT.search(t) or EMAIL_PAT.search(t) or PHONE_PAT.search(t))


# === 브랜드 별칭/오인식 교정 ===
BRAND_ALIASES = {
    r"\bbibigo\b": "비비고",
    r"\bkibigo\b": "비비고",  # k↔b 오인식 보정
    r"\bcj\b": "CJ",
    r"\bnongshim\b": "농심",
    r"\bsamyang\b": "삼양",
    r"\borion\b": "오리온",
}
KNOWN_BRANDS_HANGUL = {
    "비비고",
    "농심",
    "삼양",
    "오리온",
    "빙그레",
    "오뚜기",
    "팔도",
    "동원",
    "롯데",
    "청정원",
    "대상",
    "풀무원",
    "샘표",
    "CJ",
}


def _normalize_brand_token(s: str) -> Optional[str]:
    t = _norm(s).lower()
    t = re.sub(r"\.(com|co|kr|net|io)\b", "", t)
    for pat, canon in BRAND_ALIASES.items():
        if re.search(pat, t):
            return canon
    k = _norm(s)
    if k in KNOWN_BRANDS_HANGUL:
        return k
    return None


# === 아이템 힌트 단어 ===
FOOD_HINTS = [
    "만두",
    "왕교자",
    "교자",
    "라면",
    "국",
    "김",
    "떡",
    "스낵",
    "소시지",
    "두부",
    "볶음",
    "김치",
    "과자",
    "초코",
    "우유",
    "요거트",
]


# --- 전처리: 업스케일 + HSV-CLAHE + 언샵 ---
def _unsharp_mask(bgr, sigma=0.8, amount=0.6, threshold=0):
    blur = cv2.GaussianBlur(bgr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharp = cv2.addWeighted(bgr, 1.0 + amount, blur, -amount, 0)
    if threshold > 0:
        low_contrast_mask = np.absolute(bgr - blur) < threshold
        np.copyto(sharp, bgr, where=low_contrast_mask)
    return sharp


def _preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    img = img_bgr
    h, w = img.shape[:2]
    target_long_min = 1280
    target_long_max = 1800
    long_edge = max(h, w)
    if long_edge < target_long_min:
        scale = target_long_min / long_edge
        img = cv2.resize(img, (int(w * scale), int(h * scale)), cv2.INTER_CUBIC)
    elif long_edge > target_long_max:
        scale = target_long_max / long_edge
        img = cv2.resize(img, (int(w * scale), int(h * scale)), cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hch, sch, vch = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    vch = clahe.apply(vch)
    hsv = cv2.merge([hch, sch, vch])
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    img = _unsharp_mask(img, sigma=0.8, amount=0.6)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _init_ocr_engine():
    """PaddleOCR 3.x: GPU 우선, 실패 시 CPU로 폴백."""
    global OCR_ENGINE
    if OCR_ENGINE is not None:
        return
    if PaddleOCR is None:
        raise RuntimeError(
            "paddleocr 미설치. `pip install paddleocr opencv-python-headless`"
        )
    try:
        OCR_ENGINE = PaddleOCR(lang="korean", use_angle_cls=True, device="gpu")
        print("[OCR] init ok: lang=korean, device=gpu")
    except Exception as e:
        print(f"[OCR] GPU 초기화 실패, CPU로 재시도: {e}")
        OCR_ENGINE = PaddleOCR(lang="korean", use_angle_cls=True, device="cpu")
        print("[OCR] init ok: lang=korean, device=cpu")


def _ocr_lines(img_bgr):
    _init_ocr_engine()
    proc = _preprocess_for_ocr(img_bgr)
    result = OCR_ENGINE.ocr(proc, cls=True)
    if not result or not any(page for page in result):
        result = OCR_ENGINE.ocr(img_bgr, cls=True)

    lines = []
    if not result:
        return lines

    for page in result or []:
        if not page:
            continue
        for item in page or []:
            try:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                box = item[0]
                if isinstance(item[1], (list, tuple)):
                    text = item[1][0]
                    conf = (
                        float(item[1][1])
                        if len(item[1]) > 1 and item[1][1] is not None
                        else 0.0
                    )
                else:
                    text = item[1]
                    conf = (
                        float(item[2]) if len(item) > 2 and item[2] is not None else 0.0
                    )
                if not box:
                    continue
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                cx = sum(xs) / max(1, len(xs))
                cy = sum(ys) / max(1, len(ys))
                h = max(1.0, (max(ys) - min(ys)))
                lines.append(
                    {
                        "text": str(text).strip(),
                        "conf": conf,
                        "cx": cx,
                        "cy": cy,
                        "h": h,
                    }
                )
            except Exception:
                continue
    return lines


def _detect_panel(
    lines: List[dict], full_text: str
) -> Tuple[bool, float, Dict[str, Any]]:
    signals = {"keywords": [], "density": 0.0, "line_count": len(lines)}
    hits = []
    lower = full_text.lower()
    for kw in PANEL_KEYWORDS:
        if kw in full_text or kw.lower() in lower:
            hits.append(kw)
    signals["keywords"] = hits
    kw_score = min(1.0, len(hits) / 3.0)
    if lines:
        hs = [d["h"] for d in lines]
        median_h = sorted(hs)[len(hs) // 2]
        small_lines = sum(1 for d in lines if d["h"] <= median_h * 0.9)
        density = small_lines / max(1, len(lines))
    else:
        density = 0.0
    signals["density"] = density
    score = 0.7 * kw_score + 0.3 * density
    return (score >= 0.45), score, signals


# 보관방식 추출
COLD_REGEXES = [
    (re.compile(r"-\s*1?8\s*°?c|영하\s*1?8"), "냉동", 0.95),
    (re.compile(r"냉동"), "냉동", 0.9),
    (re.compile(r"0\s*~\s*1?\d\s*°?c"), "냉장", 0.8),
    (re.compile(r"냉장"), "냉장", 0.8),
    (re.compile(r"실온|상온|서늘한\s*곳"), "상온", 0.7),
]


def _extract_storage(text: str) -> Tuple[Optional[str], float, List[str]]:
    t = _norm(text)
    evid = []
    best = (None, 0.0)
    for rgx, label, score in COLD_REGEXES:
        if rgx.search(t):
            evid.append(label)
            if score > best[1]:
                best = (label, score)
    return best[0], best[1], list(set(evid))


LABEL_VALUE_PATTERNS = {
    "item": re.compile(r"(제품명|제품의\s*명칭|품목명|식품의\s*유형)\s*[:：]\s*(.+)"),
    "brand": re.compile(r"(제조원|판매원|수입원)\s*[:：]\s*([^\n]+)"),
}


def _extract_by_label(lines: List[dict], key: str) -> Optional[str]:
    pat = LABEL_VALUE_PATTERNS.get(key)
    if not pat:
        return None
    for ln in lines:
        m = pat.search(ln["text"])
        if m:
            val = m.group(2).strip()
            if len(val) >= 2:
                return val
    return None


def _largest_text_candidate(lines: List[dict]) -> Optional[str]:
    if not lines:
        return None
    sorted_lines = sorted(lines, key=lambda d: d["h"], reverse=True)
    for ln in sorted_lines[:12]:
        t = (ln["text"] or "").strip()
        if any(lbl in t for lbl in BLOCKLIST_LABELS):
            continue
        if _looks_like_contact_or_url(t):
            continue
        if len(re.sub(r"[^\w가-힣]", "", t)) < 2:
            continue
        return t
    return None


def _extract_item(lines: List[dict]) -> Tuple[Optional[str], float, List[str]]:
    v = _extract_by_label(lines, "item")
    if v:
        return _norm(v), 0.8, ["label:item"]
    # 힌트 단어 우선(큰 글자 우선)
    hint_lines = []
    for ln in lines:
        t = _norm(ln["text"])
        if any(h in t for h in FOOD_HINTS) and not _looks_like_contact_or_url(t):
            hint_lines.append((ln["h"], t))
    if hint_lines:
        hint_lines.sort(key=lambda x: x[0], reverse=True)
        return hint_lines[0][1], 0.7, ["hint:food-keyword"]
    v2 = _largest_text_candidate(lines)
    if v2:
        return _norm(v2), 0.6, ["large-font"]
    return None, 0.0, []


def _extract_brand(lines: List[dict]) -> Tuple[Optional[str], float, List[str]]:
    v = _extract_by_label(lines, "brand")
    if v:
        return _norm(v), 0.75, ["label:brand"]
    sorted_lines = sorted(lines, key=lambda d: d["h"], reverse=True)
    for ln in sorted_lines[:15]:
        t = (ln["text"] or "").strip()
        if _looks_like_contact_or_url(t):
            continue
        canon = _normalize_brand_token(t)
        if canon:
            return canon, 0.70, ["logo:alias-or-hangul"]
    return None, 0.0, []


def _ai_extract_from_images(file_storages: List[Any]) -> Dict[str, Any]:
    results = []
    for fs in file_storages:
        arr = np.frombuffer(fs.read(), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            results.append(
                {
                    "image_id": getattr(fs, "filename", "unknown"),
                    "error": "invalid image",
                    "has_ingredient_panel": False,
                    "panel_absent_but_resolved": False,
                    "fields": {"item_name": None, "brand": None, "storage": None},
                }
            )
            continue

        lines = _ocr_lines(img)
        full_text = "\n".join(ln["text"] for ln in lines)

        has_panel, panel_score, signals = _detect_panel(lines, full_text)
        storage, s_conf, s_e = _extract_storage(full_text)
        item, i_conf, i_e = _extract_item(lines)
        brand, b_conf, b_e = _extract_brand(lines)

        fields = {
            "item_name": {"value": item, "confidence": i_conf, "evidence": i_e},
            "brand": {"value": brand, "confidence": b_conf, "evidence": b_e},
            "storage": {"value": storage, "confidence": s_conf, "evidence": s_e},
        }
        panel_absent_but_resolved = (not has_panel) and all(
            v["value"] for v in fields.values()
        )
        results.append(
            {
                "image_id": getattr(fs, "filename", "unknown"),
                "has_ingredient_panel": has_panel,
                "panel_score": round(panel_score, 3),
                "panel_absent_but_resolved": panel_absent_but_resolved,
                "fields": fields,
                "missing_fields": [k for k, v in fields.items() if not v["value"]],
                "signals": signals,
            }
        )

    def pick(field: str):
        best = {"value": None, "confidence": 0.0, "evidence": []}
        for r in results:
            f = r["fields"].get(field, {})
            if f.get("value") and f.get("confidence", 0.0) >= best["confidence"]:
                best = f
        return best

    merged = {
        "item_name": pick("item_name"),
        "brand": pick("brand"),
        "storage": pick("storage"),
    }
    return {"results": results, "session_merged": merged}


# =========================
# 공용: 간단 응답 포맷터
# =========================
def _format_labels_response(
    raw: Dict[str, Any], verbose: bool = False
) -> Dict[str, Any]:
    merged = raw.get("session_merged", {}) or {}
    out = {
        "labels": {
            "brand": {
                "value": merged.get("brand", {}).get("value"),
                "confidence": merged.get("brand", {}).get("confidence", 0.0),
            },
            "item_name": {
                "value": merged.get("item_name", {}).get("value"),
                "confidence": merged.get("item_name", {}).get("confidence", 0.0),
            },
            "storage": {
                "value": merged.get("storage", {}).get("value"),
                "confidence": merged.get("storage", {}).get("confidence", 0.0),
            },
        },
        "images": [],
    }
    for r in raw.get("results", []):
        rec = {
            "image_id": r.get("image_id"),
            "brand": r["fields"]["brand"],
            "item_name": r["fields"]["item_name"],
            "storage": r["fields"]["storage"],
        }
        if verbose:
            rec["debug"] = {
                "panel_score": r.get("panel_score"),
                "signals": r.get("signals"),
                "missing_fields": r.get("missing_fields"),
                "panel_absent_but_resolved": r.get("panel_absent_but_resolved"),
            }
        out["images"].append(rec)
    return out


# =========================
# Routes
# =========================
@app.before_request
def log_request():
    print(f">>> {request.method} {request.path}")


@app.errorhandler(404)
def handle_404(e):
    return (
        jsonify({"error": "Not Found", "path": request.path, "method": request.method}),
        404,
    )


@app.route("/")
def home():
    return "엔드포인트: POST /labels/extract  |  POST /freshness/classify  |  POST /process (action=ai-write|fresh-check)  |  GET /health"


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "engine": ENGINE["type"],
            "model_path": MODEL_PATH,
            "image_size": IMAGE_SIZE,
            "class_names": CLASS_NAMES,
            "preprocessor_default": PREPROCESSOR,
            "gpu": {
                "available_providers": ENGINE.get("available_providers"),
                "using_providers": ENGINE.get("using_providers"),
                "gpu_enabled": any(
                    p in (ENGINE.get("using_providers") or [])
                    for p in ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
                ),
                "use_trt": ENGINE.get("use_trt"),
                "use_cuda": ENGINE.get("use_cuda"),
            },
            "input_shape": ENGINE.get("input_shape"),
            "layout": ENGINE.get("layout"),
        }
    )


# ---- 라벨 추출 (OCR+KIE)
@app.route("/labels/extract", methods=["POST"])
def labels_extract():
    if "files" not in request.files:
        return jsonify({"error": "No files field. Use key 'files'."}), 400
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "Empty file list."}), 400
    if len(files) > MAX_FILES:
        return jsonify({"error": f"Too many files. Max {MAX_FILES}."}), 400
    if cv2 is None or PaddleOCR is None:
        return (
            jsonify(
                {
                    "error": "OCR modules not installed. pip install paddleocr opencv-python-headless"
                }
            ),
            500,
        )

    for f in files:
        try:
            f.stream.seek(0)
        except Exception:
            pass

    raw = _ai_extract_from_images(files)
    verbose = request.args.get("verbose") == "1"
    return jsonify(_format_labels_response(raw, verbose=verbose))


# ---- 신선도 분류
@app.route("/freshness/classify", methods=["POST"])
def freshness_classify():
    timing = request.args.get("timing") == "1"
    debug = request.args.get("debug") == "1"
    pre_mode = (request.args.get("pre") or PREPROCESSOR).lower()

    if "files" not in request.files:
        return jsonify({"error": "No files field. Use key 'files'."}), 400
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "Empty file list."}), 400
    if len(files) > MAX_FILES:
        return jsonify({"error": f"Too many files. Max {MAX_FILES}."}), 400

    t0 = perf_counter()
    try:
        imgs = [_read_image_to_numpy(f) for f in files]
    except Exception as ex:
        return jsonify({"error": f"Failed to read image: {repr(ex)}"}), 400
    t1 = perf_counter()

    batch_np = preprocess_batch(np.stack(imgs, axis=0), pre_mode)
    t2 = perf_counter()
    probs = _predict_meet_fresh(batch_np)
    t3 = perf_counter()

    labels = [_label_from_row(row) for row in probs]
    final_label = _aggregate_with_spoiled_rule(labels)

    resp: Dict[str, Any] = {
        "final_label": final_label,
        "results": labels,
        "count": len(labels),
    }
    if debug:
        resp["probs"] = [
            {CLASS_NAMES[i]: float(row[i]) for i in range(len(CLASS_NAMES))}
            for row in probs
        ]
        resp["pre_mode"] = pre_mode
        resp["layout"] = ENGINE.get("layout")
    if timing:
        resp["timing_sec"] = {
            "load_resize": round(t1 - t0, 4),
            "preprocess": round(t2 - t1, 4),
            "inference": round(t3 - t2, 4),
            "total": round(t3 - t0, 4),
        }
    return jsonify(resp)


# ---- 통합 엔드포인트: action=ai-write | fresh-check
@app.route("/process", methods=["POST"])
def process_action():
    action = (request.form.get("action") or request.args.get("action") or "").lower()
    if action == "ai-write":
        return labels_extract()
    if action == "fresh-check":
        return freshness_classify()
    return jsonify({"error": "Invalid action. Use 'ai-write' or 'fresh-check'."}), 400


# =========================
# Entry
# =========================
if __name__ == "__main__":
    # 예) set USE_TRT=1, set USE_CUDA=0, set PREPROCESSOR=none|raw01|efficientnet, set WARMUP=1
    app.run(host="0.0.0.0", port=8000, debug=True)
