import json
import glob

paths = [
    "datasets/train_data/*.json",
    "datasets/val_data/*.json"
]

files = []
for path in paths:
    files.extend(glob.glob(path))

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Cargo Area 제거
    data["shapes"] = [
        s for s in data["shapes"]
        if "Cargo Area" not in s.get("label", "")
    ]

    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

print(f"완료: {len(files)}개 파일 처리")