#!/usr/bin/env python
"""
LabelMe JSON → 사람 crop(JPG) + train/val txt + cls_data.yaml 생성
usage:
  python gen_action_crops_split.py --src CV_Train --out action_crops --val-ratio 0.2
"""

import json, cv2, argparse, random
from pathlib import Path
from tqdm import tqdm

LABEL2ID = {"standing": 0, "sitting": 1, "lying": 2, "throwing": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def find_img_path(jfile: Path, image_path_str: str) -> Path:
    p = Path(image_path_str)
    if p.is_absolute() and p.exists():
        return p
    if (jfile.parent / p).exists():
        return jfile.parent / p
    cand = jfile.parent.parent / "Images" / p.name
    if cand.exists():
        return cand
    for ext in (".png", ".jpg", ".jpeg"):
        cand = jfile.with_suffix(ext)
        if cand.exists():
            return cand
    return None

def main(src_dir, out_dir, val_ratio):
    src = Path(src_dir)
    out = Path(out_dir)
    crop_dir = out / "images"
    crop_dir.mkdir(parents=True, exist_ok=True)

    train_txt = out / "train.txt"
    val_txt = out / "val.txt"
    yaml_path = out / "cls_data.yaml"

    all_lines = []
    skipped = 0
    json_files = sorted(src.rglob("*.json"))

    for j in tqdm(json_files, desc="Cropping"):
        meta = json.loads(j.read_text())
        img_path = find_img_path(j, meta.get("imagePath", ""))
        if img_path is None:
            skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        for i, shp in enumerate(meta.get("shapes", [])):
            (x1, y1), (x2, y2) = shp["points"]
            crop = img[int(y1):int(y2), int(x1):int(x2)]
            if crop.size == 0:
                continue

            cname = f"{j.stem}_{i}.jpg"
            crop_path = crop_dir / cname
            cv2.imwrite(str(crop_path), crop)
            label = LABEL2ID[shp["label"]]
            all_lines.append(f"{crop_path.as_posix()} {label}")

    # Shuffle & Split
    random.seed(42)
    random.shuffle(all_lines)
    n_val = int(len(all_lines) * val_ratio)
    val_lines = all_lines[:n_val]
    train_lines = all_lines[n_val:]

    train_txt.write_text("\n".join(train_lines))
    val_txt.write_text("\n".join(val_lines))

    # Write YAML
    yaml_path.write_text(f"""\
        train: {train_txt.resolve()}
        val: {val_txt.resolve()}
        nc: {len(LABEL2ID)}
        names: {list(ID2LABEL.values())}
        """)

    print(f"\n✅ Saved: {len(train_lines)} train / {len(val_lines)} val crops ➜ {crop_dir}")
    print(f"📝 train.txt / val.txt / cls_data.yaml written")
    print(f"⚠️ Skipped JSONs or images: {skipped}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset/CV_Train", help="LabelMe root dir")
    ap.add_argument("--out", default="action_crops", help="Output dir")
    ap.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    args = ap.parse_args()
    main(args.src, args.out, args.val_ratio)
