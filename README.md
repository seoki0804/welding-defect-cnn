# Welding Defect Classification CNN

방사선 투과 검사(RT) 이미지로 **용접 결함 4종을 자동 분류**하는 CNN 모델입니다.

## 배경

선박 수리 도크 현장에서 선체 재용접과 청수 탱크 천공 보수 작업의 비파괴 검사를 참관한 경험에서 출발했습니다.

방사선 투과 검사는 용접부 내부의 미세한 Crack이나 Porosity를 필름으로 확인하는데, 판독이 전적으로 검사관의 숙련도에 의존하고 시간이 오래 걸립니다. 자동 분류 모델이 검사관을 보조하면 선박 수리 품질과 속도를 개선할 수 있다고 판단했습니다.

## 분류 대상

Crack · Porosity · Excess Penetration · No Defect (RIAWELC 데이터셋, 4 class)

## 결과

**테스트 정확도 77.67%.** 다만 특정 클래스의 탐지 성능이 낮았습니다.

용접부 검사는 사용자 안전과 직결되므로, 이 정확도로는 **실무 적용이 불가하다고 판단**했습니다. 모델을 배포하지 않는다는 결론까지가 이 프로젝트의 산출물입니다.

## 아키텍처

3-Block 커스텀 CNN. 제한된 데이터와 하드웨어에서 과적합을 억제하고 일반화 성능을 끌어올리는 데 집중했습니다.

| 설계 | 의도 |
| --- | --- |
| GPU 데이터 증강 (RandomFlip·RandomRotation·RandomZoom) | 증강을 모델 첫 단에 통합해 CPU 병목 제거 |
| Conv Block 3개 (64 → 128 → 256) | 계층적 특징 추출. 각 블록은 Conv2D(3x3) + BatchNorm + ReLU + MaxPool(2x2) + Dropout |
| GlobalAveragePooling2D | Flatten 대비 파라미터 대폭 축소 — 총 439,300개 (학습가능 437,892개) |
| L2 규제 + 블록별 Dropout | 과적합 억제 |

입력 224×224×1, 출력 4-class.

## 개발 환경 제약과 해결

고성능 클라우드 GPU가 아닌 **Apple Silicon Mac 로컬 환경**에서 진행했습니다. 이 과정에서 겪은 문제와 해결이 이 프로젝트의 실질적인 내용입니다.

| 문제 | 원인 | 해결 |
| --- | --- | --- |
| PyFunc 그래프 실행 오류 | `ImageDataGenerator` 가 tensorflow-metal과 비호환 | `tf.keras.utils.image_dataset_from_directory` 로 데이터 로더 교체 |
| 14 에포크에서 학습 중단, `InvalidArgumentError` | `.DS_Store` 등 숨김 파일과 손상 이미지를 이미지로 읽으려 시도 | 진단 코드로 train/valid/test 전체 스캔 후 비정상 파일 제거 |
| 배치 사이즈 64에서 OOM | Mac 통합 GPU의 제한된 VRAM | 배치 사이즈 32로 조정 |
| 방대한 원본 데이터셋 | 로컬 자원으로 전량 학습 불가 | 샘플링 전략으로 데이터셋 규모 축소 |

## 파일

```
welding_defect_cnn.ipynb        모델 개발 노트북 (50 epoch)
report_welding_defect.docx      개발 보고서
```

## 기술 스택

TensorFlow/Keras · tensorflow-metal (Apple Silicon GPU) · Python
