# 🥬 Meat Freshness & OCR API Server

**Flask + ONNXRuntime + PaddleOCR** 기반의 통합 서버입니다.  
단일 컨테이너에서 이미지 신선도 분류와 식품 성분표 OCR 라벨 추출을 제공합니다.

---

## 🔗 Endpoints

| Method | Path                  | Description                                    |
|--------|-----------------------|------------------------------------------------|
| GET    | `/health`             | 엔진/모델/환경 상태 확인                       |
| POST   | `/labels/extract`     | 성분표 OCR + 항목명/브랜드/보관방식 추출       |
| POST   | `/freshness/classify` | 이미지 신선도 분류 (Fresh/Half-Fresh/Spoiled)  |
| POST   | `/process`            | 통합 엔드포인트 (`action=ai-write` or `fresh-check`) |

---

## 🚀 Quickstart

### 1. Docker 실행

GPU 환경:
```bash
docker run -d -p 8000:8000 \
  -e USE_CUDA=1 -e USE_TRT=0 \
  -v $(pwd)/model:/app/model \
  --name meat-server your_image:latest
