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

def kfold_split(lines, k):
    random.seed(42)
    random.shuffle(lines)
    fold_size = len(lines) // k
    folds = [lines[i * fold_size:(i + 1) * fold_size] for i in range(k - 1)]
    folds.append(lines[(k - 1) * fold_size:])
    return folds

def main_kfold(src_dir, out_dir, k_folds):
    src = Path(src_dir)
    out = Path(out_dir)
    crop_dir = out / "images"
    crop_dir.mkdir(parents=True, exist_ok=True)

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

    folds = kfold_split(all_lines, k_folds)

    for i in range(k_folds):
        fold_out = out / f"fold{i+1}"
        fold_out.mkdir(parents=True, exist_ok=True)
        train_txt = fold_out / "train.txt"
        val_txt = fold_out / "val.txt"
        yaml_path = fold_out / "cls_data.yaml"

        val_lines = folds[i]
        train_lines = [line for j, fold in enumerate(folds) if j != i for line in fold]

        train_txt.write_text("\n".join(train_lines))
        val_txt.write_text("\n".join(val_lines))
        yaml_path.write_text(f"""\
train: {train_txt.resolve()}
val: {val_txt.resolve()}
nc: {len(LABEL2ID)}
names: {list(ID2LABEL.values())}
""")

    print(f"\n✅ K-Fold split done with {k_folds} folds. Skipped JSONs or images: {skipped}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset", help="LabelMe root dir")
    ap.add_argument("--out", default="action_crops_kfold", help="Output dir")
    ap.add_argument("--kfold", type=int, default=5, help="Number of K folds")
    args = ap.parse_args()
    main_kfold(args.src, args.out, args.kfold)