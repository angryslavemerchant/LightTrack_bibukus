"""
tracking/prepare_coco_patches.py

Crops template + search patches from COCO train2017.
Uses ultralytics YOLODataset for image/label discovery.
~50% of images are randomly sampled. Single 2.0× search crop per image.

Expected local layout (ultralytics default):
    data/coco/images/train2017/   — raw images
    data/coco/labels/train2017/   — YOLO-format .txt labels

Output layout:
    <OUTPUT_ROOT>/
        template/   — short-side square crop, resized to 128×128
        search/     — 2.0× template side, resized to 256×256

Filename convention: <image_stem>.jpg
"""

import os
import random
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.downloads import download

# =============================================================================
# CONFIG
# =============================================================================

COCO_DIR      = Path('data/coco')
IMG_DIR       = COCO_DIR / 'images' / 'train2017'
OUTPUT_ROOT   = 'data/coco_patches'

SAMPLE_FRACTION = 0.5
TEMPLATE_SIZE   = 128
SEARCH_SIZE     = 256
SEARCH_MULT     = 2.0
MIN_BBOX_AREA   = 400   # minimum bbox area in pixels²
SEED            = 42

# =============================================================================
# HELPERS
# =============================================================================

def crop_square(img: np.ndarray, cx: float, cy: float, half: float) -> np.ndarray:
    """Zero-padded square crop centred at (cx, cy) with given half-side length."""
    h, w = img.shape[:2]
    size = int(round(half * 2))
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    x1, y1 = x0 + size, y0 + size

    pad_l = max(0, -x0);    pad_t = max(0, -y0)
    pad_r = max(0, x1 - w); pad_b = max(0, y1 - h)

    crop = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if any((pad_l, pad_t, pad_r, pad_b)):
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r,
                                  cv2.BORDER_CONSTANT, value=0)
    return crop

# =============================================================================
# MAIN
# =============================================================================

def download_coco():
    """Download COCO train2017 images and YOLO labels if not already present."""
    if not IMG_DIR.exists():
        print('Downloading COCO train2017 images (~18 GB) ...')
        download('http://images.cocodataset.org/zips/train2017.zip',
                 dir=COCO_DIR / 'images', unzip=True, delete=True)

    labels_dir = COCO_DIR / 'labels' / 'train2017'
    if not labels_dir.exists():
        misplaced = COCO_DIR / 'coco' / 'labels'
        if misplaced.exists():
            # zip extracted one level too deep — move it into place
            print('Fixing label directory location ...')
            shutil.move(str(misplaced), str(COCO_DIR / 'labels'))
        else:
            print('Downloading COCO YOLO labels ...')
            download('https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip',
                     dir=COCO_DIR.parent, unzip=True, delete=True)


def prepare_patches():
    download_coco()

    tmpl_dir   = os.path.join(OUTPUT_ROOT, 'template')
    search_dir = os.path.join(OUTPUT_ROOT, 'search')
    os.makedirs(tmpl_dir,   exist_ok=True)
    os.makedirs(search_dir, exist_ok=True)

    print(f'Loading dataset from {IMG_DIR} ...')
    ds = YOLODataset(
        img_path=str(IMG_DIR),
        data=dict(nc=80, names={i: str(i) for i in range(80)}),
        augment=False,
    )

    random.seed(SEED)
    n_total  = len(ds)
    n_sample = int(n_total * SAMPLE_FRACTION)
    indices  = random.sample(range(n_total), n_sample)
    print(f'  {n_total} total images → sampling {n_sample} ({SAMPLE_FRACTION*100:.0f}%)')

    saved = skipped = 0
    for idx in tqdm(indices):
        label   = ds.labels[idx]
        im_file = ds.im_files[idx]
        bboxes  = label['bboxes']   # (N, 4) normalized xywh

        if len(bboxes) == 0:
            skipped += 1
            continue

        img = cv2.imread(im_file)
        if img is None:
            skipped += 1
            continue
        img_h, img_w = img.shape[:2]

        # Pick largest bbox above area threshold
        areas = bboxes[:, 2] * bboxes[:, 3] * img_w * img_h
        valid = np.where(areas >= MIN_BBOX_AREA)[0]
        if len(valid) == 0:
            skipped += 1
            continue
        best        = valid[np.argmax(areas[valid])]
        xc, yc, bw, bh = bboxes[best]

        cx          = xc * img_w
        cy          = yc * img_h
        half_tmpl   = min(bw * img_w, bh * img_h) / 2
        half_search = half_tmpl * SEARCH_MULT

        fname = Path(im_file).stem + '.jpg'

        tmpl = crop_square(img, cx, cy, half_tmpl)
        cv2.imwrite(os.path.join(tmpl_dir, fname),
                    cv2.resize(tmpl, (TEMPLATE_SIZE, TEMPLATE_SIZE),
                               interpolation=cv2.INTER_LINEAR))

        search = crop_square(img, cx, cy, half_search)
        cv2.imwrite(os.path.join(search_dir, fname),
                    cv2.resize(search, (SEARCH_SIZE, SEARCH_SIZE),
                               interpolation=cv2.INTER_LINEAR))

        saved += 1

    print(f'\nDone. {saved} pairs saved, {skipped} skipped.')
    print(f'Output: {OUTPUT_ROOT}')


if __name__ == '__main__':
    prepare_patches()
