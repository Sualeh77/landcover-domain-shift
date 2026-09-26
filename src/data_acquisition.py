"""GEE data download: Sentinel-2 composites + WorldCover labels."""

import ee
import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyproj import Transformer
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────

GEE_PROJECT = "landcover-shift"
CHIP_SIZE = 256
SCALE = 10  # meters per pixel
CHIP_METERS = CHIP_SIZE * SCALE  # 2560m

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
WORLDCOVER = "ESA/WorldCover/v200"
S2_BANDS = ["B2", "B3", "B4", "B8"]
REFLECTANCE_SCALE = 10000.0

NODATA_REFLECTANCE = -1.0
NODATA_LABEL = 255
MAX_EMPTY_FRAC = 0.05
NUM_CLASSES = 7
CLASS_NAMES = ["tree", "shrub", "grass", "crop", "built", "bare", "water"]

WORLDCOVER_FROM = [10, 20, 30, 40, 50, 60, 80, 70, 90, 95, 100]
WORLDCOVER_TO = [0, 1, 2, 3, 4, 5, 6, 255, 2, 0, 5]

SCL_MASK_VALUES = [0, 1, 3, 8, 9, 10, 11]


@dataclass
class AOIConfig:
    name: str
    west: float
    south: float
    east: float
    north: float
    date_start: str
    date_end: str
    n_chips: int
    crs: str


SOURCE_AOI = AOIConfig(
    name="source",
    west=-93.80, south=41.75, east=-93.15, north=42.20,
    date_start="2021-07-01", date_end="2021-08-31",
    n_chips=300, crs="EPSG:32615",
)

TARGET_AOI = AOIConfig(
    name="target",
    west=1.87, south=13.30, east=2.38, north=13.69,
    date_start="2021-10-01", date_end="2021-11-30",
    n_chips=300, crs="EPSG:32631",
)


# ─── GEE helpers ─────────────────────────────────────────────────────

def initialize_gee():
    ee.Initialize(project=GEE_PROJECT)


def _mask_s2_clouds(image):
    scl = image.select("SCL")
    mask = ee.Image.constant(1)
    for val in SCL_MASK_VALUES:
        mask = mask.And(scl.neq(val))
    return image.updateMask(mask)


def build_s2_composite(aoi: AOIConfig) -> ee.Image:
    bbox = ee.Geometry.Rectangle([aoi.west, aoi.south, aoi.east, aoi.north])

    collection = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(bbox)
        .filterDate(aoi.date_start, aoi.date_end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))
        .map(_mask_s2_clouds)
    )

    def _select_and_scale(img):
        bands = img.select(S2_BANDS).divide(REFLECTANCE_SCALE).clamp(0, 1)
        ndvi = bands.normalizedDifference(["B8", "B4"]).rename("NDVI").clamp(0, 1)
        return bands.addBands(ndvi)

    composite = collection.map(_select_and_scale).median()
    # Unmask so permanently clouded pixels get -1 instead of masked
    composite = composite.unmask(NODATA_REFLECTANCE)
    return composite.clip(bbox)


def get_worldcover_remapped(aoi: AOIConfig) -> ee.Image:
    bbox = ee.Geometry.Rectangle([aoi.west, aoi.south, aoi.east, aoi.north])
    wc = ee.ImageCollection(WORLDCOVER).first().clip(bbox)
    remapped = wc.select("Map").remap(WORLDCOVER_FROM, WORLDCOVER_TO, defaultValue=255)
    return remapped.rename("label").toUint8()


# ─── Grid generation ────────────────────────────────────────────────

def generate_chip_grid(aoi: AOIConfig, seed: int = 42) -> list[dict]:
    transformer = Transformer.from_crs("EPSG:4326", aoi.crs, always_xy=True)
    utm_west, utm_south = transformer.transform(aoi.west, aoi.south)
    utm_east, utm_north = transformer.transform(aoi.east, aoi.north)

    all_chips = []
    y = utm_south
    while y + CHIP_METERS <= utm_north:
        x = utm_west
        while x + CHIP_METERS <= utm_east:
            all_chips.append({
                "utm_west": x, "utm_south": y,
                "utm_east": x + CHIP_METERS, "utm_north": y + CHIP_METERS,
                "crs": aoi.crs,
            })
            x += CHIP_METERS
        y += CHIP_METERS

    logger.info(f"{aoi.name}: {len(all_chips)} potential chips in grid, sampling {aoi.n_chips}")
    rng = random.Random(seed)
    if len(all_chips) <= aoi.n_chips:
        return all_chips
    return rng.sample(all_chips, aoi.n_chips)


# ─── Chip download ──────────────────────────────────────────────────

def _compute_pixels_with_retry(request_params: dict, max_retries: int = 3) -> bytes:
    for attempt in range(max_retries):
        try:
            return ee.data.computePixels(request_params)
        except Exception as e:
            err_str = str(e)
            if any(s in err_str for s in ["429", "Too many", "RESOURCE_EXHAUSTED"]):
                wait = 2 ** attempt + random.random()
                logger.warning(f"Rate limited (attempt {attempt+1}), retrying in {wait:.1f}s")
                time.sleep(wait)
            elif attempt < max_retries - 1:
                time.sleep(1)
            else:
                raise
    raise RuntimeError(f"Failed after {max_retries} retries")


def download_chip(
    composite: ee.Image,
    labels: ee.Image,
    chip_info: dict,
    chip_idx: int,
    output_dir: Path,
    max_retries: int = 3,
) -> bool:
    img_path = output_dir / "images" / f"chip_{chip_idx:03d}.npy"
    lbl_path = output_dir / "labels" / f"chip_{chip_idx:03d}.npy"

    if img_path.exists() and lbl_path.exists():
        return True

    transform = {
        "scaleX": SCALE, "shearX": 0, "translateX": chip_info["utm_west"],
        "shearY": 0, "scaleY": -SCALE, "translateY": chip_info["utm_north"],
    }
    grid = {
        "dimensions": {"width": CHIP_SIZE, "height": CHIP_SIZE},
        "affineTransform": transform,
        "crsCode": chip_info["crs"],
    }

    # Download image bands
    img_request = {
        "expression": composite,
        "fileFormat": "NUMPY_NDARRAY",
        "grid": grid,
    }
    arr = _compute_pixels_with_retry(img_request, max_retries)
    band_names = ["B2", "B3", "B4", "B8", "NDVI"]
    image = np.stack([arr[b].astype(np.float32) for b in band_names], axis=0)

    # Check empty pixel fraction
    empty_mask = image[0] <= NODATA_REFLECTANCE
    empty_frac = empty_mask.sum() / empty_mask.size
    if empty_frac > MAX_EMPTY_FRAC:
        logger.warning(f"Chip {chip_idx}: {empty_frac:.1%} empty pixels, skipping")
        return False

    # Replace any remaining nodata with 0
    image = np.where(image <= NODATA_REFLECTANCE, 0.0, image)

    # Download labels
    lbl_request = {
        "expression": labels,
        "fileFormat": "NUMPY_NDARRAY",
        "grid": grid,
    }
    lbl_arr = _compute_pixels_with_retry(lbl_request, max_retries)
    label = lbl_arr["label"].astype(np.uint8)

    np.save(img_path, image)
    np.save(lbl_path, label)
    return True


# ─── Domain download orchestrator ───────────────────────────────────

def download_domain(
    aoi: AOIConfig,
    output_base: Path,
    max_workers: int = 10,
    seed: int = 42,
) -> dict:
    initialize_gee()

    composite = build_s2_composite(aoi)
    labels = get_worldcover_remapped(aoi)
    chips = generate_chip_grid(aoi, seed=seed)

    img_dir = output_base / "images"
    lbl_dir = output_base / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    failed_indices = []

    def _do_download(args):
        idx, chip_info = args
        try:
            return idx, download_chip(composite, labels, chip_info, idx, output_base)
        except Exception as e:
            logger.error(f"Chip {idx} failed: {e}")
            return idx, False

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_do_download, (i, chip)): i
            for i, chip in enumerate(chips)
        }
        with tqdm(total=len(chips), desc=f"Downloading {aoi.name}") as pbar:
            for future in as_completed(futures):
                idx, ok = future.result()
                if ok:
                    success += 1
                else:
                    failed_indices.append(idx)
                pbar.update(1)

    stats = {
        "total": len(chips),
        "success": success,
        "failed": len(failed_indices),
        "failed_indices": sorted(failed_indices),
    }
    logger.info(f"{aoi.name}: {stats}")
    return stats


# ─── Splits ──────────────────────────────────────────────────────────

def create_splits(
    source_dir: Path,
    target_dir: Path,
    source_val_frac: float = 0.15,
    n_target_eval: int = 60,
    seed: int = 42,
) -> dict:
    # Source: spatial block split (hold out southernmost rows as val)
    # Re-generate the grid to get UTM coordinates for each chip index
    source_grid = generate_chip_grid(SOURCE_AOI, seed=seed)
    source_chips = sorted([
        int(p.stem.split("_")[1])
        for p in (source_dir / "images").glob("chip_*.npy")
    ])

    # Sort chips by utm_south to identify spatial blocks
    chip_positions = []
    for idx in source_chips:
        if idx < len(source_grid):
            chip_positions.append((idx, source_grid[idx]["utm_south"]))
    chip_positions.sort(key=lambda x: x[1])

    # Southernmost fraction as val (spatially contiguous block)
    n_val = int(len(chip_positions) * source_val_frac)
    val_indices = sorted([idx for idx, _ in chip_positions[:n_val]])
    train_indices = sorted([idx for idx, _ in chip_positions[n_val:]])

    source_splits = {"val": val_indices, "train": train_indices}
    with open(source_dir / "splits.json", "w") as f:
        json.dump(source_splits, f, indent=2)

    # Target: random split (spatial leakage less critical for eval-only)
    rng = random.Random(seed)
    target_chips = sorted([
        int(p.stem.split("_")[1])
        for p in (target_dir / "images").glob("chip_*.npy")
    ])
    rng.shuffle(target_chips)
    target_splits = {
        "eval": sorted(target_chips[:n_target_eval]),
        "train": sorted(target_chips[n_target_eval:]),
    }
    with open(target_dir / "splits.json", "w") as f:
        json.dump(target_splits, f, indent=2)

    return {"source": source_splits, "target": target_splits}


# ─── Loading and stats ───────────────────────────────────────────────

def load_chip(domain_dir: Path, idx: int):
    image = np.load(domain_dir / "images" / f"chip_{idx:03d}.npy")
    label = np.load(domain_dir / "labels" / f"chip_{idx:03d}.npy")
    return image, label


def compute_dataset_stats(domain_dir: Path, chip_indices: list[int]) -> dict:
    n_bands = len(S2_BANDS) + 1  # +NDVI
    running_sum = np.zeros(n_bands, dtype=np.float64)
    running_sq = np.zeros(n_bands, dtype=np.float64)
    n_pixels = 0
    class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    for idx in chip_indices:
        image, label = load_chip(domain_dir, idx)
        # image: (5, 256, 256)
        pixels = image.reshape(n_bands, -1)  # (5, N)
        running_sum += pixels.sum(axis=1)
        running_sq += (pixels ** 2).sum(axis=1)
        n_pixels += pixels.shape[1]
        for c in range(NUM_CLASSES):
            class_counts[c] += (label == c).sum()

    mean = running_sum / n_pixels
    std = np.sqrt(running_sq / n_pixels - mean ** 2)
    total_labeled = class_counts.sum()
    class_freq = class_counts / total_labeled if total_labeled > 0 else class_counts

    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "class_counts": class_counts.tolist(),
        "class_freq": class_freq.tolist(),
        "n_chips": len(chip_indices),
        "n_pixels": n_pixels,
    }
