# Dataset README

이 프로젝트의 `datasets` 폴더는 철스크랩 이미지 segmentation 학습과 PQ 평가를 위한 데이터셋을 관리합니다.

전체 흐름은 LabelMe JSON과 이미지 원본을 입력으로 받아, Cargo Area 기준 crop, polygon 필터링, 클래스 리맵, COCO instance 변환, YOLO segmentation label 변환 순서로 진행됩니다.

## 폴더 구조

```text
datasets/
  train_data/             # 원본 train LabelMe JSON + 이미지
  val_data/               # 원본 val LabelMe JSON + 이미지

  train_cropped/          # Cargo Area 기준으로 crop된 train JSON + 이미지
  val_cropped/            # Cargo Area 기준으로 crop된 val JSON + 이미지

  train_data_filtered/    # 너무 작은 polygon을 제거한 train LabelMe JSON
  val_data_filtered/      # 너무 작은 polygon을 제거한 val LabelMe JSON

  train_remapped/         # 원본 상세 라벨을 19개 학습 클래스로 매핑한 train JSON
  val_remapped/           # 원본 상세 라벨을 19개 학습 클래스로 매핑한 val JSON

  images/
    train/                # YOLO 학습용 train 이미지
    val/                  # YOLO 검증용 val 이미지

  labels/
    train/                # YOLO segmentation train txt label
    val/                  # YOLO segmentation val txt label

  annotations/
    instances_train.json  # COCO instance 형식 train annotation
    instances_val.json    # COCO instance 형식 val annotation

  classes.txt             # 학습 클래스 목록
  data.yaml               # YOLO 학습용 dataset 설정 파일
```

## 전처리 단계

`python run_data_preprocessing.py`를 실행하면 아래 순서로 일괄 처리됩니다.

1. `0_build_cargo_dataset.py`
   - `datasets/train_data`, `datasets/val_data`를 입력으로 사용합니다.
   - `"73. Cargo Area"` polygon을 찾아 해당 영역 기준으로 이미지를 crop합니다.
   - 결과를 `datasets/train_cropped`, `datasets/val_cropped`에 저장합니다.

2. `0_remove_cargo.py`
   - crop된 JSON에서 `"Cargo Area"` annotation을 제거합니다.

3. `1_remove_small_filter.py`
   - YOLO 입력 크기 기준으로 너무 작은 polygon을 제거합니다.
   - 결과 JSON은 `train_data_filtered`, `val_data_filtered`에 저장합니다.
   - crop 이미지는 `images/train`, `images/val`로 복사합니다.

4. `2_remap_labelme_exact.py`
   - 원본 상세 라벨을 학습용 통합 라벨로 매핑합니다.
   - 결과를 `train_remapped`, `val_remapped`에 저장합니다.

5. `3_annotations_to_instances.py`
   - remapped LabelMe JSON을 COCO instance 형식으로 변환합니다.
   - `annotations/instances_train.json`, `annotations/instances_val.json`을 생성합니다.

6. `4_labels_yolo.py`
   - COCO instance annotation을 YOLO segmentation txt label로 변환합니다.
   - `labels/train`, `labels/val`을 생성합니다.
   - `classes.txt`, `data.yaml`도 함께 생성합니다.

## 학습 클래스

현재 데이터셋은 19개 클래스를 사용합니다.

```text
handler
rebar
structure steel
mixed steel
heavy iron
panel
square pipe
mesh
trash
pipe
small pipe
vehicle
plastic
machine
drum
LPG GAS cylinder
beam
Fan
Guillotine
```

YOLO 학습 시에는 `datasets/data.yaml`을 사용합니다.

```bash
yolo task=segment mode=train data=datasets/data.yaml model=yolo11s-seg.pt epochs=500 imgsz=1024
```

## PQ 평가 관련 산출물

PQ 평가는 `python run_pq_eval.py`로 실행합니다.

관련 경로는 다음과 같습니다.

```text
GT_visualization/              # GT mask 시각화 이미지
output/                        # YOLO 예측 mask 시각화 이미지
PQ/
  panoptic_test.json           # GT panoptic annotation
  panoptic_test/               # GT panoptic PNG mask
  panoptic_predictions.json    # prediction panoptic annotation
  panoptic_pred/               # prediction panoptic PNG mask
```

PQ 평가 단계는 `datasets/annotations/instances_val.json`, `datasets/images/val`, YOLO 예측 결과인 `runs/segment/predict/labels`를 사용합니다.

## 주의 사항

- `train_data`, `val_data`에는 같은 basename을 가진 이미지와 JSON을 함께 넣어야 합니다.
- LabelMe JSON의 `imagePath`, `imageWidth`, `imageHeight` 값은 후속 변환에서 사용됩니다.
- `0_build_cargo_dataset.py`는 Cargo Area 기준 crop만 수행하며, crop 후 작은 polygon 면적 필터는 적용하지 않습니다.
- `1_remove_small_filter.py`는 YOLO resize 기준의 작은 polygon 필터를 별도로 수행합니다.
- `run_data_preprocessing.py`를 다시 실행하면 일부 중간 산출물 폴더가 재생성될 수 있으므로, 수동 수정한 중간 파일이 있다면 먼저 백업해야 합니다.
