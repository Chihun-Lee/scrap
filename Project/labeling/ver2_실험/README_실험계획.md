# ver2 라벨링 기준 연구 실험 (2·3·4순위) — 실행 가이드

작성: 2026-07-08. ITIV 7/7 메일 회신(`메일회신2/회신초안_20260708.md`) 근거 실험 모음.
**모든 크기 기준은 1280×1280 입력 환산** (원본 3840×2160 → s=1/3, 즉 1280 기준 1px = 원본 3px).

## 이미 완료 (CPU, 본 폴더에 결과 있음)

| 파일 | 내용 |
|---|---|
| `instance_size_stats.py` / `_summary.md`, `_train.csv` | remapped(ver1 필터 통과분 138,896개) 기준 크기 분포 |
| `instance_size_stats_raw.py` / `_raw_summary.md` | **원본 미필터(417,218개) 기준** — 메일 회신 수치의 출처 |
| `exp5_stride4_survival.py` / `exp5_survival_summary.md`, `exp5_instances.csv.gz`, `exp5_panels/` | **stride4 GT 생존 전수 시뮬레이션(50.1만 개, 2026-07-14)** — ITIV 확인요청 3번 대응. ultralytics polygon2mask(fillPoly@1280→resize 1/4) 재현. 결론: 완전 소실은 드묾, 문제는 형상 충실도(원본 18px 미만 IoU<0.5 과반)와 세장형 조각남(두께 2px@1280 미만 74.6% 분절). `--scale-factor 0.5`(증강 최악조건) 결과는 `_s0.5` 접미사 |

핵심 수치(원본 train): sqrt(area)@1280 <8px = 48.0% / <10px = 59.2%, 면적비 <0.007%≈8px 컷.
bbox 짧은변 <16px@1280 = 278,322개 → ver1의 8px@640 필터 제거 수와 정확히 일치(교차 검증 완료).

## GPU 실험 (클러스터에서 실행 — 미실행 상태)

실행 전 준비:
```bash
# 맥북에서 동기화
cd ~/Code && ./sync-to-cluster.sh scrap
# 클러스터에서
conda activate scrap
pip install -U ultralytics   # yolo26+imgsz1280 이슈 픽스 포함 최신(>=8.4.90) 필수
cd ~/scrap/Project/labeling/ver2_실험
```

### 1) 데이터셋 변형 생성 — `prepare_yolo_dataset.py`
원본 LabelMe(train_data/val_data, 미필터) → 89→19 리맵(LABEL_MAP 재사용) → 컷오프/세장형 예외/RDP 단순화 적용 → YOLO seg 포맷(`datasets_exp/<name>/`). 이미지는 심볼릭 링크(용량 0).
exp2/exp3 스크립트가 필요 변형을 자동 생성하므로 단독 실행은 선택.

### 2) 2순위 재검증 — `exp2_train_sweep.py` (목표: 7/24 공유)
컷오프 {무필터, 8, 10, 12, 16}px × YOLO26x-seg @1280. 결과 → `exp2_results.csv`.
잠정 제안치는 10px(원본 30px, 0.011%) — 1024 스윕 최적점(원본 29px, 축소비 0.55)을 원본픽셀로 전사한 하한.
입력픽셀 기준 전사 시나리오면 최적이 16px(원본 48px)까지 올라갈 수 있어 스윕 범위가 이 구간(24~48px)을 커버함.
⚠️ 판독 주의: 과거 1024 스윕은 각 조건이 자기 필터 적용된 val로 평가됐을 가능성 → exp2에서는 **공통 val(무필터 base의 val 라벨) 기준 재평가를 병행**해 비교 왜곡을 제거할 것.
```bash
# 스모크 테스트 (작은 모델, 짧게 — 파이프라인 검증용)
python exp2_train_sweep.py --model yolo11s-seg.pt --cuts 8 --epochs 5
# 본 실험 (조건당 yolo26x@1280 100ep — 조건당 대략 하루 예상, 총 5조건)
nohup python exp2_train_sweep.py --epochs 100 > exp2.log 2>&1 &
```
판독: mask mAP50이 최고이면서 세장형 클래스(rebar/small pipe/pipe) AP가 유지되는 컷 채택.

### 2b) 세장형 예외 정책 ablation — `exp6_exception_ablation.py` (ITIV 7/14 확인요청 대응, exp2와 함께 7/24 공유)
컷오프 10px 고정, 예외 정책 3종 비교: `cut10_noexc`(예외 없음) / `cut10`(긴변≥24px, 현행) / `cut10_w2`(+shoelace 두께≥2px@1280=원본 6px, exp5 정책 C2).
exp5 시뮬레이션 근거(두께 원본 6px 미만 세장형은 GT가 절반 이상 조각남 → 학습 노이즈 가능성)를 실제 학습으로 검증.
데이터셋 변형 3종은 로컬에서 생성 완료(kept: noexc 204,135 / cut10 232,978 / w2 222,029 — exp5 정책표 예측과 ±0.1% 일치). 세장형 클래스별 mask AP50을 공통 val 기준으로 CSV에 기록.
```bash
nohup python exp6_exception_ablation.py --epochs 100 > exp6.log 2>&1 &
```
판독: cut10_w2 ≥ cut10 이면(특히 rebar/small pipe AP) "긴변 예외에 두께 조건 추가"를 라벨링 기준에 반영.

### 3) 3순위 그리드 — `exp3_imgsz_points.py` (목표: 스크리닝 7/31, 최종 8/4)
imgsz {960,1280,1600,1920} × RDP eps {0, 4, 8}px@1280 (컷오프는 8px 고정). 결과 → `exp3_results.csv`.
```bash
nohup python exp3_imgsz_points.py --epochs 60 > exp3.log 2>&1 &          # 12조합 스크리닝
python exp3_imgsz_points.py --imgsz 1280 1600 --eps 4 --epochs 300       # 상위 조합 확정
```
- eps 4px@1280 = 원본 12px = proto cell 1개 (이론상 무손실 단순화 상한)
- 주의: eps는 절대값이라 소형 객체일수록 형상 왜곡이 큼(예: 27pt→3pt, 면적 -37%) — 이것 자체가 측정 대상 효과이며, 소형 recall 변화를 exp4와 교차 확인할 것
- 1920은 연산 4×(960 대비) — peak_vram_gb / infer_ms 컬럼으로 균형점 판단

### 4) 4순위 클래스별 최소크기 — `exp4_min_size_analysis.py` (목표: 중간 8/21, 최종 9/4)
2·3순위 best 가중치로 val 추론 → 클래스×크기구간 recall → `exp4_recall_by_size.csv`.
```bash
python exp4_min_size_analysis.py --weights runs/exp2_cut8_yolo26x-seg_e100/weights/best.pt --variant base
```
판독: 클래스별로 recall이 급락하는 최소 구간 = 그 클래스의 "학습 유의미 최소 크기".
rebar/small pipe(중앙값 ~10px@1280)는 별도 하한 예상 — 세장형 예외 규칙(bbox 긴변≥24px 유지)의 근거 확인.

## 주의사항
- OOM 시: `--batch` 를 4→2로 명시(기본 -1 AutoBatch). 이미지당 인스턴스 ~200개라 AutoBatch 추정보다 실사용이 큼.
- yolo26x-seg.pt는 첫 실행 시 자동 다운로드(사내망 프록시 주의 — 안 되면 맥북에서 받아 rsync).
- 학습 무의미 컷은 "라벨 삭제"가 아니라 학습용 사본에서의 필터임 — 원본 라벨은 보존(6/23 방침: 라벨은 보수적으로, 필터는 코드로).
