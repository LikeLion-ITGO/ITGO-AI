# Meat Freshness Classifier

EfficientNetB0를 기반으로 육류 신선도를 **Fresh / Half-Fresh / Spoiled** 3가지로 분류하는 이미지 분류 모델입니다.  
TensorFlow/Keras로 구현하였고, 데이터 증강과 파인튜닝 과정을 포함합니다.

---

## 데이터셋
- **출처**: [Meat Freshness Image Dataset (Kaggle)](https://www.kaggle.com/datasets/vinayakshanawad/meat-freshness-image-dataset)  
- **클래스 구성**:
  1. Fresh  
  2. Half-Fresh  
  3. Spoiled  
- **이미지 크기**: 원본 이미지를 224×224로 리사이즈  
- **전처리 과정**:
  - `tf.keras.applications.efficientnet.preprocess_input` 사용
  - 데이터 증강: 수평·수직 플립, 회전, 확대, 대비 변화, 밝기 변화
- **데이터 분할 비율**:
  - 학습: 검증 = 약 80% : 20%

## 모델 구조
- **Base Model**: EfficientNetB0 (ImageNet 사전학습 가중치, top layer 제거)

- **학습 단계**:
1. **초기 학습**: Base 모델 전체 동결, Adam(lr=0.001), 10 epochs
2. **파인튜닝**: 마지막 20개 레이어만 학습, Adam(lr=1e-5), 20 epochs
- **손실 함수**: Categorical Crossentropy  
- **평가지표**: Accuracy

## 평가 지표 (Validation set 예시)
| Class       | Precision | Recall | F1-score |
|-------------|-----------|--------|----------|
| Fresh       | 0.97      | 0.96   | 0.96     |
| Half-Fresh  | 0.95      | 0.95   | 0.95     |
| Spoiled     | 0.97      | 1.00   | 0.99     |
| **Accuracy**|            **0.96**           |
