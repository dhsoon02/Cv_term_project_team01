import json, random, argparse, shutil, sys
from pathlib import Path
from PIL import Image
from math import ceil

LABEL2ID = {"person": 0}

def convert_one(jfile: Path, img_out: Path, lbl_out: Path):
    meta = json.loads(jfile.read_text())
    img_rel = Path(meta["imagePath"])

    if img_rel.is_absolute():
        src_img = img_rel
    else:
        cand1 = jfile.parent.parent / "Images" / img_rel.name
        cand2 = jfile.parent / img_rel
        cand3 = jfile.with_suffix(".png")
        for c in (cand1, cand2, cand3):
            if c.exists():
                src_img = c
                break
        else:
            print(f"[WARN] image not found: {img_rel} for {jfile}", file=sys.stderr)
            return

    dst_img = img_out / src_img.name
    shutil.copy2(src_img, dst_img)

    w, h = Image.open(src_img).size
    yolo_lines = []
    for sh in meta["shapes"]:
        (x1, y1), (x2, y2) = sh["points"]
        xc, yc = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        cls_id = 0
        yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    (lbl_out / f"{dst_img.stem}.txt").write_text("\n".join(yolo_lines))

def convert_kfold(json_files, dst_base: Path, k: int):
    total = len(json_files)
    fold_size = ceil(total / k)

    for i in range(k):
        fold_name = f"fold{i+1}"
        fold_dir = dst_base / fold_name
        img_tr = fold_dir / "images" / "train"
        img_val = fold_dir / "images" / "val"
        lbl_tr = fold_dir / "labels" / "train"
        lbl_val = fold_dir / "labels" / "val"

        for p in [img_tr, img_val, lbl_tr, lbl_val]:
            p.mkdir(parents=True, exist_ok=True)

        val_start = i * fold_size
        val_end = min(val_start + fold_size, total)
        val_set = set(json_files[val_start:val_end])

        skipped = 0
        for jf in json_files:
            img_dir, lbl_dir = (img_val, lbl_val) if jf in val_set else (img_tr, lbl_tr)
            before = len(list(img_dir.glob("*.png")))
            convert_one(jf, img_dir, lbl_dir)
            after = len(list(img_dir.glob("*.png")))
            if after == before:
                skipped += 1

        print(f"[Fold {i+1}] Train: {len(list(img_tr.glob('*.png')))}, Val: {len(list(img_val.glob('*.png')))}, Skipped: {skipped}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset")
    ap.add_argument("--dst", default="dataset_yolo_only_person_kfold")
    ap.add_argument("--kfold", type=int, default=5)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    k = args.kfold

    json_files = sorted(src.rglob("*.json"))
    if not json_files:
        print(f"[ERROR] No JSON found under {src}.", file=sys.stderr)
        sys.exit(1)

    random.seed(42)
    random.shuffle(json_files)

    convert_kfold(json_files, dst, k)

if __name__ == "__main__":
    main()