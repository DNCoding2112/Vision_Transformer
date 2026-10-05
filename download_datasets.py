"""
download_datasets.py
====================
Download and prepare 3 AI-image detection datasets on the SPIT GPU cluster.

Datasets (all via Kaggle API):
  1. CIFAKE          ~110 MB   birdy654/cifake-real-and-ai-generated-synthetic-images
  2. AI-vs-Human     ~2.4 GB   alessandrasala79/ai-vs-human-generated-dataset
  3. 140k Faces      ~0.8 GB   xhlulu/140k-real-and-fake-faces

Total: ~3.3 GB.

Deepfake-Faces is intentionally NOT included:
  The Kaggle mirror (dagnelies/deepfake-faces) ships only FAKE face crops
  (no REAL counterpart). Pairing its FAKE with 140k REAL would teach the
  model a sharp-vs-blurry shortcut instead of deepfake artifacts.

Usage:
    python download_datasets.py                 # full download + normalise
    python download_datasets.py --skip_download # normalise only (raw/ already present)
    python download_datasets.py --fast          # generate dummy data, test the pipeline
"""

import os
import sys
import glob
import shutil
import random
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

RAW_DIR  = "./raw"
DATA_DIR = "./data"

# ============================================================
# Helpers
# ============================================================
def collect_images(src_dirs, exts=("*.jpg", "*.jpeg", "*.png", "*.webp")):
    """Recursively collect image paths from a list of directories."""
    paths = []
    for d in src_dirs:
        for ext in exts:
            paths += glob.glob(os.path.join(d, "**", ext), recursive=True)
    return paths


def copy_subset(paths, dst, max_n=None, seed=42):
    """Copy (up to max_n) images into dst. Renames with hash prefix to avoid collisions."""
    os.makedirs(dst, exist_ok=True)
    if max_n and len(paths) > max_n:
        rng = random.Random(seed)
        paths = rng.sample(paths, max_n)
    copied = 0
    for p in paths:
        fname = f"{abs(hash(p)) % (10**10)}_{os.path.basename(p)}"
        try:
            shutil.copy2(p, os.path.join(dst, fname))
            copied += 1
        except Exception as e:
            log.warning(f"Copy failed {p}: {e}")
    return copied


# ============================================================
# Per-dataset normalisation
# ============================================================
def normalise_cifake():
    """
    CIFAKE raw layout:
        raw/CIFAKE/train/REAL/   raw/CIFAKE/train/FAKE/
        raw/CIFAKE/test/REAL/    raw/CIFAKE/test/FAKE/
    Target:
        data/cifake/{train,test}/{REAL,FAKE}
    """
    log.info("[CIFAKE] Normalising...")
    raw = os.path.join(RAW_DIR, "CIFAKE")
    if not os.path.isdir(raw):
        log.error(f"  Raw folder missing: {raw}")
        return

    for split in ("train", "test"):
        for label in ("REAL", "FAKE"):
            src = os.path.join(raw, split, label)
            dst = os.path.join(DATA_DIR, "cifake", split, label)
            if not os.path.isdir(src):
                log.warning(f"  Missing: {src}")
                continue
            imgs = collect_images([src])
            n = copy_subset(imgs, dst)
            log.info(f"  {split}/{label}: {n:,}")


def normalise_aivsreal():
    """
    AI-vs-Human — labels come from train.csv / test.csv.
    Raw layout:
        raw/AI-vs-Human/train.csv         (columns: filename, label — auto-detected)
        raw/AI-vs-Human/test.csv
        raw/AI-vs-Human/train_data/       (flat *.jpg)
        raw/AI-vs-Human/test_data_v2/     (flat *.jpg)
    Target:
        data/aivsreal/{train,test}/{REAL,FAKE}
    Capped at 30,000 per class per split.
    """
    import csv
    log.info("[AI-vs-Human] Normalising (CSV-labels)...")
    raw = os.path.join(RAW_DIR, "AI-vs-Human")
    if not os.path.isdir(raw):
        log.error(f"  Raw folder missing: {raw}")
        return

    def read_csv(csv_path, img_dir):
        if not os.path.isfile(csv_path):
            log.warning(f"  CSV missing: {csv_path}")
            return []
        rows = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            log.info(f"  {os.path.basename(csv_path)} columns: {cols}")

            # Auto-detect filename & label columns
            fname_col = None
            label_col = None
            for c in cols:
                cl = c.lower()
                if fname_col is None and any(k in cl for k in ("file", "image", "name", "id", "path")):
                    fname_col = c
                if label_col is None and any(k in cl for k in ("label", "class", "target", "is_ai")):
                    label_col = c
            if fname_col is None: fname_col = cols[0]
            if label_col is None: label_col = cols[1]
            log.info(f"  using fname_col='{fname_col}'  label_col='{label_col}'")

            for row in reader:
                fname = (row.get(fname_col) or "").strip()
                rl    = (row.get(label_col) or "").strip()
                ll    = rl.lower()

                # Normalise to 0=REAL, 1=FAKE
                if ll in ("0", "real", "human", "false"):
                    y = 0
                elif ll in ("1", "fake", "ai", "ai_generated", "generated", "true"):
                    y = 1
                else:
                    try:
                        y = int(float(rl))
                    except Exception:
                        continue
                    y = 1 if y == 1 else 0

                # Resolve image path — CSV may store basename only, or a subpath
                base = os.path.basename(fname)
                for cand in (os.path.join(img_dir, fname),
                             os.path.join(img_dir, base)):
                    if os.path.isfile(cand):
                        rows.append((cand, y))
                        break
        return rows

    all_train = read_csv(os.path.join(raw, "train.csv"), os.path.join(raw, "train_data"))
    all_test  = read_csv(os.path.join(raw, "test.csv"),  os.path.join(raw, "test_data_v2"))
    log.info(f"  Parsed: train={len(all_train):,}  test={len(all_test):,}")

    if not all_train and not all_test:
        log.error("  Both CSVs empty — cannot proceed")
        return

    CAP = 30000
    def cap_split(rows):
        by = {0: [], 1: []}
        for p, y in rows:
            by[y].append(p)
        out = []
        for y in (0, 1):
            out += [(p, y) for p in by[y][:CAP]]
        return out

    all_train = cap_split(all_train)
    all_test  = cap_split(all_test)

    for split, rows in (("train", all_train), ("test", all_test)):
        for label_int, label_str in ((0, "REAL"), (1, "FAKE")):
            dst = os.path.join(DATA_DIR, "aivsreal", split, label_str)
            paths = [p for p, y in rows if y == label_int]
            n = copy_subset(paths, dst, max_n=CAP)
            log.info(f"  {split}/{label_str}: {n:,}")
def normalise_140k():
    """
    140k Faces raw layout:
        raw/140k-Real-and-Fake-Faces/real_vs_fake/real-vs-fake/train/real/
        raw/140k-Real-and-Fake-Faces/real_vs_fake/real-vs-fake/train/fake/
        raw/140k-Real-and-Fake-Faces/real_vs_fake/real-vs-fake/valid/real/
        raw/140k-Real-and-Fake-Faces/real_vs_fake/real-vs-fake/valid/fake/
    Target maps 'valid' -> 'test'.
    Kept in full (train 100k, valid 20k).
    """
    log.info("[140k Faces] Normalising...")
    raw = os.path.join(RAW_DIR, "140k-Real-and-Fake-Faces")
    if not os.path.isdir(raw):
        log.error(f"  Raw folder missing: {raw}")
        return

    base = os.path.join(raw, "real_vs_fake", "real-vs-fake")
    if not os.path.isdir(base):
        candidates = glob.glob(os.path.join(raw, "**", "real-vs-fake"), recursive=True)
        if candidates:
            base = candidates[0]
        else:
            log.error(f"  Cannot locate real-vs-fake/ under {raw}")
            return

    for split, src_split in (("train", "train"), ("test", "valid")):
        for label, src_label in (("REAL", "real"), ("FAKE", "fake")):
            src = os.path.join(base, src_split, src_label)
            dst = os.path.join(DATA_DIR, "faces140k", split, label)
            if not os.path.isdir(src):
                log.warning(f"  Missing: {src}")
                continue
            imgs = collect_images([src])
            n = copy_subset(imgs, dst)
            log.info(f"  {split}/{label}: {n:,}")


# ============================================================
# Summary
# ============================================================
DATASETS = ("cifake", "aivsreal", "faces140k")


def print_summary():
    log.info("\n" + "=" * 62)
    log.info("DATA DISTRIBUTION SUMMARY")
    log.info("=" * 62)

    total_train_real = total_train_fake = 0
    total_test_real  = total_test_fake  = 0
    per_ds = {}

    for ds in DATASETS:
        counts = {"train_real": 0, "train_fake": 0, "test_real": 0, "test_fake": 0}
        for split in ("train", "test"):
            for label in ("REAL", "FAKE"):
                d = os.path.join(DATA_DIR, ds, split, label)
                n = len(glob.glob(os.path.join(d, "*"))) if os.path.isdir(d) else 0
                counts[f"{split}_{label.lower()}"] = n
        per_ds[ds] = counts

        total_train_real += counts["train_real"]
        total_train_fake += counts["train_fake"]
        total_test_real  += counts["test_real"]
        total_test_fake  += counts["test_fake"]

        log.info(
            f"  {ds:<12} | train REAL={counts['train_real']:>7,} FAKE={counts['train_fake']:>7,} "
            f"| test REAL={counts['test_real']:>6,} FAKE={counts['test_fake']:>6,}"
        )

    log.info("-" * 62)
    log.info(
        f"  {'TOTAL':<12} | train REAL={total_train_real:>7,} FAKE={total_train_fake:>7,} "
        f"| test REAL={total_test_real:>6,} FAKE={total_test_fake:>6,}"
    )
    ratio = total_train_real / max(total_train_fake, 1)
    log.info(f"  {'':<12} | Train REAL/FAKE ratio: {ratio:.3f}")
    log.info("=" * 62)

    ok = True
    if total_train_real == 0 or total_train_fake == 0:
        log.error("Training set has an empty class — check data/ layout!")
        ok = False
    if total_test_real == 0 or total_test_fake == 0:
        log.error("Test set has an empty class — check data/ layout!")
        ok = False

    for ds in DATASETS:
        c = per_ds[ds]
        if sum(c.values()) == 0:
            log.error(f"Dataset '{ds}' produced 0 images — not loaded!")
            ok = False

    if ok:
        log.info("All datasets loaded. Ready to train:  python main.py --mode train")
    else:
        sys.exit(1)

    return per_ds


# ============================================================
# Fast-mode dummy generator
# ============================================================
def generate_fast_dummy(total_images=480):
    """Create a small balanced dataset for pipeline smoke-testing."""
    import numpy as np
    from PIL import Image as PILImage

    log.info(f"FAST MODE: generating {total_images} dummy images...")
    per_ds_per_split = total_images // (len(DATASETS) * 2 * 2)

    for ds in DATASETS:
        for split in ("train", "test"):
            for label in ("REAL", "FAKE"):
                d = os.path.join(DATA_DIR, ds, split, label)
                os.makedirs(d, exist_ok=True)
                for i in range(per_ds_per_split):
                    path = os.path.join(d, f"dummy_{split}_{label}_{i}.jpg")
                    if not os.path.exists(path):
                        color = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                        PILImage.fromarray(color).save(path, quality=85)
    log.info("Dummy dataset ready.")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare datasets")
    parser.add_argument("--kaggle_json", type=str, default=None,
                        help="Path to kaggle.json (optional — env vars used otherwise)")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip Kaggle download, only normalise existing raw/")
    parser.add_argument("--fast", action="store_true",
                        help="Generate dummy images instead of downloading")
    args = parser.parse_args()

    if args.fast:
        generate_fast_dummy()
        print_summary()
        sys.exit(0)

    # Kaggle credentials
    os.environ.setdefault("KAGGLE_USERNAME", "sidjain433")
    os.environ.setdefault("KAGGLE_KEY", "KGAT_4883274cd23999933c5cdb53547baddb")

    # Ensure kaggle CLI is available
    try:
        import kaggle  # noqa: F401
    except ImportError:
        log.info("Installing kaggle CLI...")
        os.system("uv pip install kaggle || pip install kaggle")

    # Download
    if not args.skip_download:
        downloads = [
            ("birdy654/cifake-real-and-ai-generated-synthetic-images", "CIFAKE",                    110),
            ("alessandrasala79/ai-vs-human-generated-dataset",         "AI-vs-Human",              2400),
            ("xhlulu/140k-real-and-fake-faces",                        "140k-Real-and-Fake-Faces", 800),
        ]
        for slug, name, size_mb in downloads:
            out_dir = os.path.join(RAW_DIR, name)
            os.makedirs(out_dir, exist_ok=True)

            existing = collect_images([out_dir])
            if len(existing) > 100:
                log.info(f"[{name}] Already downloaded ({len(existing):,} images). Skipping.")
                continue

            log.info(f"[{name}] Downloading ~{size_mb} MB ...")
            ret = os.system(f"kaggle datasets download -d {slug} -p {out_dir} --unzip")
            if ret != 0:
                log.error(f"[{name}] Download failed (exit {ret})")
            else:
                n = len(collect_images([out_dir]))
                log.info(f"[{name}] Done — {n:,} images in raw/")
    else:
        log.info("Skipping downloads (--skip_download)")

    # Normalise into data/
    normalise_cifake()
    normalise_aivsreal()
    normalise_140k()

    # Report
    print_summary()