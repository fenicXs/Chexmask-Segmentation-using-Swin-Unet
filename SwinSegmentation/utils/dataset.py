"""
Dataset loader for CheXmask database
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import cv2
from sklearn.model_selection import train_test_split


def get_mask_from_RLE(rle, height, width):
    """Decode Run-Length Encoding (RLE) to a binary mask array."""

    height = int(height)
    width = int(width)
    mask = np.zeros(height * width, dtype=np.uint8)

    def reshape_mask(flat_mask: np.ndarray) -> np.ndarray:
        """Reshape RLE-decoded flat mask into (height, width) using row-major order."""

        return flat_mask.reshape((width, height), order='F').T

    if rle is None:
        return reshape_mask(mask)

    if isinstance(rle, float):
        if np.isnan(rle):
            return mask.reshape((height, width), order='F')
        rle = str(int(rle))

    rle_str = str(rle).strip()
    if not rle_str or rle_str == '-1':
        return reshape_mask(mask)

    try:
        tokens = [int(x) for x in rle_str.split()]
    except ValueError:
        return reshape_mask(mask)

    for idx in range(0, len(tokens), 2):
        start = tokens[idx] - 1
        if start < 0 or idx + 1 >= len(tokens):
            continue
        length = tokens[idx + 1]
        end = start + length
        mask[start:end] = 1

    return reshape_mask(mask)


def load_dataset_dataframe(
    csv_path: str,
    quality_threshold: float,
    dataset_name: Optional[str] = None,
) -> pd.DataFrame:
    """Load dataset annotations and apply quality filtering."""

    df = pd.read_csv(csv_path)
    if 'Dice RCA (Mean)' in df.columns and quality_threshold is not None:
        df = df[df['Dice RCA (Mean)'] >= quality_threshold]
    df = df.reset_index(drop=True)

    if dataset_name and 'dataset' not in df.columns:
        df = df.assign(dataset=dataset_name)

    return df


def _normalize_split_label(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    if isinstance(label, float) and np.isnan(label):
        return None

    value = str(label).strip().lower()
    if value in {'train', 'training'}:
        return 'train'
    if value in {'val', 'valid', 'validation', 'dev'}:
        return 'val'
    if value in {'test', 'testing', 'holdout'}:
        return 'test'
    return None


def detect_image_id_column(df: pd.DataFrame) -> str:
    """Heuristically determine which column stores the image identifier."""

    preferred = [
        'image_id',
        'dicom_id',
        'sop_instance_uid',
        'Path',
        'image_path',
        'image',
        'filename',
        'Image Index',
        'uid',
    ]

    for column in preferred:
        if column in df.columns:
            return column

    return df.columns[0]


def _indices_from_split_map(
    df: pd.DataFrame,
    id_col: str,
    split_map: Mapping[str, str],
    dataset_name: str,
    seed: int,
    fallback_val_ratio: float,
) -> Tuple[List[int], List[int], List[int]]:
    """Map split labels to dataset indices with sensible fallbacks."""

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    id_series = df[id_col].astype(str)

    for idx, identifier in enumerate(id_series):
        split = split_map.get(identifier)
        if split is None:
            split = split_map.get(Path(identifier).name)

        if split is None and 'Path' in df.columns:
            alt = str(df.iloc[idx]['Path'])
            split = split_map.get(alt) or split_map.get(Path(alt).name)

        normalized = _normalize_split_label(split)
        if normalized == 'train':
            train_indices.append(idx)
        elif normalized == 'val':
            val_indices.append(idx)
        elif normalized == 'test':
            test_indices.append(idx)

    if not val_indices and train_indices:
        rng = np.random.default_rng(seed)
        shuffled = np.array(train_indices)
        rng.shuffle(shuffled)

        fallback_count = max(1, int(round(len(train_indices) * fallback_val_ratio)))
        if fallback_count >= len(train_indices) and len(train_indices) > 1:
            fallback_count = len(train_indices) - 1

        val_indices = shuffled[:fallback_count].tolist()
        train_indices = shuffled[fallback_count:].tolist()

        print(
            f"[{dataset_name}] Created validation split of {len(val_indices)} samples"
            f" using fallback ratio {fallback_val_ratio}."
        )

    if not test_indices and val_indices:
        test_indices = list(val_indices)
        print(
            f"[{dataset_name}] No explicit test split; reusing validation set"
            f" ({len(test_indices)} samples)."
        )

    train_indices.sort()
    val_indices.sort()
    test_indices.sort()

    print(
        f"[{dataset_name}] Split sizes -> Train: {len(train_indices)},"
        f" Val: {len(val_indices)}, Test: {len(test_indices)}"
    )

    return train_indices, val_indices, test_indices


def _load_vindr_rel_paths(txt_path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with txt_path.open() as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            rel_path = parts[0]
            image_id = Path(rel_path).name
            mapping[image_id] = rel_path
    return mapping


def _compute_vindr_splits(
    df: pd.DataFrame,
    id_col: str,
    settings: Mapping[str, str],
    seed: int,
    fallback_val_ratio: float,
) -> Tuple[List[int], List[int], List[int], Dict[str, Sequence[str]]]:
    train_txt = settings.get('train_split')
    test_txt = settings.get('test_split')
    if not train_txt or not test_txt:
        raise ValueError("VinDr split configuration missing 'train_split' or 'test_split'.")

    train_map = _load_vindr_rel_paths(Path(train_txt))
    test_map = _load_vindr_rel_paths(Path(test_txt))

    available_ids = set(df[id_col].astype(str))
    train_ids = [img_id for img_id in train_map if img_id in available_ids]
    test_ids = [img_id for img_id in test_map if img_id in available_ids]

    unique_train = np.array(sorted(set(train_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_train)

    val_ratio = settings.get('val_ratio', fallback_val_ratio)
    if len(unique_train) > 1:
        val_count = max(1, int(round(len(unique_train) * val_ratio)))
        if val_count >= len(unique_train):
            val_count = len(unique_train) - 1
    else:
        val_count = 0

    val_ids = set(unique_train[:val_count])
    train_ids_final = set(unique_train[val_count:])

    split_map: Dict[str, str] = {}
    split_map.update({img_id: 'train' for img_id in train_ids_final})
    split_map.update({img_id: 'val' for img_id in val_ids})
    split_map.update({img_id: 'test' for img_id in sorted(set(test_ids))})

    indices = _indices_from_split_map(df, id_col, split_map, 'VinDr-CXR', seed, fallback_val_ratio)

    path_lookup: Dict[str, Sequence[str]] = {}
    for source in (train_map, test_map):
        for img_id, rel_path in source.items():
            if img_id in available_ids:
                path_lookup.setdefault(img_id, [])
                if rel_path not in path_lookup[img_id]:
                    path_lookup[img_id].append(rel_path)

    return (*indices, {'path_lookup': path_lookup})


def _load_mimic_filename_map(map_path: Path) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    with map_path.open() as handle:
        for line in handle:
            rel = line.strip()
            if not rel:
                continue
            dicom_id = Path(rel).stem
            mapping.setdefault(dicom_id, [])
            if rel not in mapping[dicom_id]:
                mapping[dicom_id].append(rel)
    return mapping


def _compute_mimic_splits(
    df: pd.DataFrame,
    id_col: str,
    settings: Mapping[str, str],
    seed: int,
    fallback_val_ratio: float,
) -> Tuple[List[int], List[int], List[int], Dict[str, Sequence[str]]]:
    split_csv = settings.get('split_csv')
    map_file = settings.get('filename_map')
    if not split_csv or not map_file:
        raise ValueError("MIMIC split configuration requires 'split_csv' and 'filename_map'.")

    split_series = (
        pd.read_csv(split_csv, usecols=['dicom_id', 'split'])
        .set_index('dicom_id')['split']
        .map(_normalize_split_label)
    )

    split_map = split_series.to_dict()
    indices = _indices_from_split_map(df, id_col, split_map, 'MIMIC-CXR-JPG', seed, fallback_val_ratio)

    path_lookup = _load_mimic_filename_map(Path(map_file))
    return (*indices, {'path_lookup': path_lookup})


def _chexpert_variants(path_str: str) -> List[str]:
    path_str = path_str.strip()
    variants = [path_str]
    if path_str.startswith('CheXpert-v1.0/'):
        trimmed = path_str[len('CheXpert-v1.0/') :]
        variants.append(trimmed)
    else:
        variants.append(f"CheXpert-v1.0/{path_str}")
    # Remove duplicates while preserving order
    seen = set()
    unique_variants = []
    for item in variants:
        if item not in seen:
            unique_variants.append(item)
            seen.add(item)
    return unique_variants


def _compute_chexpert_splits(
    df: pd.DataFrame,
    id_col: str,
    settings: Mapping[str, str],
    seed: int,
    fallback_val_ratio: float,
) -> Tuple[List[int], List[int], List[int], Dict[str, Sequence[str]]]:
    train_csv = settings.get('train_csv')
    valid_csv = settings.get('valid_csv')
    if not train_csv or not valid_csv:
        raise ValueError("CheXpert split configuration requires 'train_csv' and 'valid_csv'.")

    split_map: Dict[str, str] = {}

    train_df = pd.read_csv(train_csv, usecols=['Path'])
    for path in train_df['Path']:
        for variant in _chexpert_variants(path):
            split_map[variant] = 'train'

    valid_df = pd.read_csv(valid_csv, usecols=['Path'])
    for path in valid_df['Path']:
        for variant in _chexpert_variants(path):
            split_map[variant] = 'val'

    indices = _indices_from_split_map(df, id_col, split_map, 'CheXpert', seed, fallback_val_ratio)

    # CheXpert CSV already points to file paths; no additional lookup needed
    return (*indices, {})


def resolve_dataset_splits(
    config,
    df: pd.DataFrame,
    id_col: str,
) -> Tuple[List[int], List[int], List[int], Dict[str, Sequence[str]]]:
    split_config = getattr(config, 'SPLIT_CONFIG', {})
    dataset_settings = split_config.get(config.DATASET_NAME, {})
    fallback_val_ratio = getattr(config, 'VAL_RATIO', 0.1)

    if config.DATASET_NAME == 'VinDr-CXR':
        return _compute_vindr_splits(df, id_col, dataset_settings, config.SEED, fallback_val_ratio)
    if config.DATASET_NAME == 'MIMIC-CXR-JPG':
        return _compute_mimic_splits(df, id_col, dataset_settings, config.SEED, fallback_val_ratio)
    if config.DATASET_NAME == 'CheXpert':
        return _compute_chexpert_splits(df, id_col, dataset_settings, config.SEED, fallback_val_ratio)

    train_indices, val_indices, test_indices = create_data_splits(
        config.CSV_PATH,
        train_ratio=getattr(config, 'TRAIN_RATIO', 0.7),
        val_ratio=getattr(config, 'VAL_RATIO', 0.15),
        test_ratio=getattr(config, 'TEST_RATIO', 0.15),
        quality_threshold=config.QUALITY_THRESHOLD,
        seed=config.SEED,
        dataframe=df,
    )
    return train_indices, val_indices, test_indices, {}


def _append_unique(container: List[Path], candidate: Path) -> None:
    if candidate not in container:
        container.append(candidate)


def _vin_dr_candidate_paths(base: Path, rel: str) -> List[Path]:
    rel = str(rel)
    extensions = ('.dicom', '.dcm', '.jpg', '.jpeg', '.png')
    candidates: List[Path] = []

    rel_variants = [rel]
    rel_alt = rel.replace('train_jpeg', 'train').replace('test_jpeg', 'test').replace('val_jpeg', 'val')
    if rel_alt != rel:
        rel_variants.append(rel_alt)

    for variant in rel_variants:
        rel_path = Path(variant)
        _append_unique(candidates, base / rel_path)

        suffix = rel_path.suffix.lower()
        if suffix in {'.jpg', '.jpeg', '.png'}:
            for ext in ('.dicom', '.dcm'):
                _append_unique(candidates, (base / rel_path).with_suffix(ext))
        elif not suffix:
            for ext in extensions:
                _append_unique(candidates, base / Path(f"{variant}{ext}"))

    return candidates


def _chexpert_candidate_paths(base: Path, rel: str) -> List[Path]:
    candidates: List[Path] = []
    for variant in _chexpert_variants(rel):
        rel_path = Path(variant)
        _append_unique(candidates, base / rel_path)
        _append_unique(candidates, base / Path('CheXpert-v1.0') / rel_path)
    return candidates


class CheXmaskDataset(Dataset):
    """
    PyTorch Dataset for CheXmask segmentation data
    
    Args:
        csv_path: Path to the CSV file containing annotations
        image_base_path: Base path where the actual X-ray images are stored
        dataset_name: Name of the dataset (e.g., 'VinDr-CXR', 'MIMIC-CXR-JPG')
        quality_threshold: Minimum Dice RCA (Mean) score to include samples
        transform: Optional transforms to apply
        mode: 'train', 'val', or 'test'
    """
    
    def __init__(
        self,
        csv_path,
        image_base_path=None,
        dataset_name='',
        quality_threshold=0.7,
        transform=None,
        mode='train',
        dataframe=None,
        path_lookup: Optional[Mapping[str, Sequence[str]]] = None,
        image_roots: Optional[Mapping[str, str]] = None,
    ):
        super().__init__()
        
        self.csv_path = csv_path
        self.image_base_path = Path(image_base_path) if image_base_path else None
        self.image_roots = {str(key): Path(value) for key, value in (image_roots or {}).items()}
        self.dataset_name = str(dataset_name)
        self.quality_threshold = quality_threshold
        self.transform = transform
        self.mode = mode
        self.path_lookup: Dict[str, List[str]] = {}
        if path_lookup:
            for key, value in path_lookup.items():
                key_str = str(key)
                if isinstance(value, (list, tuple)):
                    self.path_lookup[key_str] = [str(v) for v in value]
                else:
                    self.path_lookup[key_str] = [str(value)]
        
        # Load the CSV file or use provided dataframe
        if dataframe is not None:
            print(f"Using provided dataframe with {len(dataframe)} samples")
            self.df = dataframe
        else:
            print(f"Loading {dataset_name} dataset from {csv_path}")
            self.df = pd.read_csv(csv_path)
            print(f"Loaded {len(self.df)} samples")
        
        # Filter by quality threshold
        if 'Dice RCA (Mean)' in self.df.columns:
            self.df = self.df[self.df['Dice RCA (Mean)'] >= quality_threshold]
            print(f"After quality filtering (>= {quality_threshold}): {len(self.df)} samples")
        
        # Reset index
        self.df = self.df.reset_index(drop=True)

        # Get the column name for image ID (varies by dataset)
        self.image_id_col = self._get_image_id_column()

    def _get_image_id_column(self):
        """Get the appropriate image ID column name for the dataset"""
        return detect_image_id_column(self.df)

    def _resolve_base_path(self, dataset_name: str) -> Path:
        dataset_key = str(dataset_name)
        if self.image_roots:
            base = self.image_roots.get(dataset_key)
            if base is None:
                raise KeyError(f"No image root configured for dataset '{dataset_key}'.")
            return base
        if self.image_base_path is None:
            raise ValueError("Image base path is not configured for dataset loading.")
        return self.image_base_path
    
    def __len__(self):
        return len(self.df)
    
    def _load_image(self, image_id, dataset_name, relpath=None):
        """Load chest X-ray image from disk."""

        dataset_key = str(dataset_name)
        base_path = self._resolve_base_path(dataset_key)

        candidates: List[Path] = []

        if relpath is not None and not (pd.isna(relpath) if isinstance(relpath, float) else False):
            relpath_str = str(relpath).strip()
            if relpath_str:
                if dataset_key == 'VinDr-CXR':
                    for candidate in _vin_dr_candidate_paths(base_path, relpath_str):
                        _append_unique(candidates, candidate)
                elif dataset_key == 'CheXpert':
                    for candidate in _chexpert_candidate_paths(base_path, relpath_str):
                        _append_unique(candidates, candidate)
                else:
                    _append_unique(candidates, base_path / relpath_str)

        image_id_str = str(image_id)

        if dataset_key == 'VinDr-CXR':
            lookup_values = self.path_lookup.get(image_id_str)
            if not lookup_values:
                lookup_values = self.path_lookup.get(Path(image_id_str).name)
            if lookup_values:
                for rel in lookup_values:
                    for candidate in _vin_dr_candidate_paths(base_path, rel):
                        _append_unique(candidates, candidate)
            else:
                for candidate in _vin_dr_candidate_paths(base_path, image_id_str):
                    _append_unique(candidates, candidate)

            for folder in ('train', 'test'):
                _append_unique(candidates, base_path / folder / image_id_str)
                _append_unique(candidates, base_path / folder / f"{image_id_str}.dicom")
                _append_unique(candidates, base_path / folder / f"{image_id_str}.dcm")

        elif dataset_key == 'MIMIC-CXR-JPG':
            lookup_values = self.path_lookup.get(image_id_str)
            if not lookup_values:
                lookup_values = self.path_lookup.get(Path(image_id_str).stem)
            if lookup_values:
                for rel in lookup_values:
                    _append_unique(candidates, base_path / rel)
            else:
                _append_unique(candidates, base_path / image_id_str)

        elif dataset_key == 'CheXpert':
            for candidate in _chexpert_candidate_paths(base_path, image_id_str):
                _append_unique(candidates, candidate)

        elif dataset_key == 'Padchest':
            _append_unique(candidates, base_path / image_id_str)

        else:
            _append_unique(candidates, base_path / image_id_str)

        img_path: Optional[Path] = None
        for candidate in candidates:
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            raise FileNotFoundError(
                f"Image not found for ID '{image_id_str}'. Tried: "
                + ", ".join(str(p) for p in candidates[:5])
                + (" ..." if len(candidates) > 5 else "")
            )

        img_path_str = str(img_path)

        if img_path_str.lower().endswith('.dicom') or img_path_str.lower().endswith('.dcm'):
            try:
                import pydicom
                from pydicom.pixel_data_handlers.util import apply_voi_lut

                dcm = pydicom.dcmread(img_path_str)

                image = apply_voi_lut(dcm.pixel_array, dcm)

                if dcm.PhotometricInterpretation == "MONOCHROME1":
                    image = np.max(image) - image

                if image.dtype != np.uint8:
                    image = image.astype(np.float64)
                    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
                    image = (image * 255).astype(np.uint8)

            except Exception as e:
                raise ValueError(f"Could not load DICOM image {img_path_str}: {e}")
        else:
            image = cv2.imread(img_path_str, cv2.IMREAD_GRAYSCALE)

            if image is None:
                raise ValueError(f"Could not load image: {img_path_str}")

        return image
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        
        # Get row from dataframe
        row = self.df.iloc[idx]
        
        # Get image ID
        image_id = str(row[self.image_id_col])
        dataset_name = row.get('dataset', self.dataset_name)
        if pd.isna(dataset_name):
            dataset_name = self.dataset_name
        image_relpath = row.get('image_relpath')
        
        # Load image
        try:
            image = self._load_image(image_id, dataset_name, image_relpath)
        except Exception as e:
            print(f"Error loading image {image_id}: {e}")
            # Return a dummy sample
            image = np.zeros((1024, 1024), dtype=np.float32)
        
        # Get mask dimensions
        height = int(row['Height'])
        width = int(row['Width'])
        
        # Decode RLE masks
        right_lung_mask = get_mask_from_RLE(row['Right Lung'], height, width)
        left_lung_mask = get_mask_from_RLE(row['Left Lung'], height, width)
        heart_mask = get_mask_from_RLE(row['Heart'], height, width)
        
        # Resize image to match mask size if needed
        if image.shape[0] != height or image.shape[1] != width:
            image = cv2.resize(image, (width, height))
        
        # Create multi-class mask
        # Class 0: Background
        # Class 1: Right Lung
        # Class 2: Left Lung
        # Class 3: Heart
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[right_lung_mask > 0] = 1
        mask[left_lung_mask > 0] = 2
        mask[heart_mask > 0] = 3
        
        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Apply transforms
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Convert to tensors
        # Add channel dimension to image
        image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
        mask = torch.from_numpy(mask).long()  # (H, W)
        
        sample = {
            'image': image,
            'mask': mask,
            'image_id': image_id,
            'height': height,
            'width': width,
            'dataset': dataset_name
        }
        
        return sample


def create_data_splits(csv_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                      quality_threshold=0.7, seed=42, dataframe: Optional[pd.DataFrame] = None):
    """
    Split dataset into train, validation, and test sets
    
    Returns:
        train_indices, val_indices, test_indices
    """
    if dataframe is None:
        df = pd.read_csv(csv_path)
        if 'Dice RCA (Mean)' in df.columns:
            df = df[df['Dice RCA (Mean)'] >= quality_threshold]
        df = df.reset_index(drop=True)
    else:
        df = dataframe.reset_index(drop=True)
    
    # Create indices
    indices = np.arange(len(df))
    
    # Split into train and temp (val + test)
    train_indices, temp_indices = train_test_split(
        indices, test_size=(val_ratio + test_ratio), random_state=seed, shuffle=True
    )
    
    # Split temp into val and test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=(1 - val_size), random_state=seed, shuffle=True
    )
    
    return train_indices, val_indices, test_indices


def get_dataloaders(config):
    """
    Create data loaders for training, validation, and testing
    
    Args:
        config: Configuration module
        
    Returns:
        train_loader, val_loader, test_loader
    """
    import albumentations as A
    
    # Define transforms
    train_transform = None
    val_transform = None
    
    if config.USE_AUGMENTATION:
        # CheXmask-Database compatible augmentation (from HybridGNet training)
        train_transform = A.Compose([
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
            # Horizontal flip (common for chest X-rays)
            A.HorizontalFlip(p=0.5) if config.AUG_CONFIG['horizontal_flip'] else A.NoOp(),
            # Rotation with proper border handling (matches CheXmask approach)
            # FIXED: Removed invalid 'value' parameter, using interpolation only
            A.Rotate(
                limit=config.AUG_CONFIG['rotation_range'],  # ±10 degrees
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT,
                interpolation=cv2.INTER_LINEAR
            ),
            # Brightness and contrast (matches CheXmask preprocessing)
            A.RandomBrightnessContrast(
                brightness_limit=config.AUG_CONFIG['brightness'],  # 0.2 = ±20%
                contrast_limit=config.AUG_CONFIG['contrast'],  # 0.2 = ±20%
                p=0.5
            ),
            # Gamma correction (CheXmask uses range 0.67-1.5, here 0.85-1.15)
            A.RandomGamma(
                gamma_limit=(85, 115),  # Albumentations expects (min%, max%)
                p=0.5
            ),
            # Gaussian noise (σ = 1/128 ≈ 0.0078 in CheXmask, using std=0.01)
            # FIXED: Latest Albumentations changed GaussNoise API
            # Use GaussianBlur or simple GaussNoise() if parameters are deprecated
            A.GaussNoise(p=0.3),  # Uses default noise parameters
        ])
    else:
        train_transform = A.Compose([
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        ])
    
    val_transform = A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
    ])

    if getattr(config, 'USE_COMPOSITE_DATASET', False):
        return _get_composite_dataloaders(config, train_transform, val_transform)

    df = load_dataset_dataframe(config.CSV_PATH, config.QUALITY_THRESHOLD)
    id_col = detect_image_id_column(df)
    train_indices, val_indices, test_indices, extras = resolve_dataset_splits(config, df, id_col)
    path_lookup = extras.get('path_lookup', {}) if extras else {}

    print(
        f"[{config.DATASET_NAME}] split -> train: {len(train_indices):,} | "
        f"val: {len(val_indices):,} | test: {len(test_indices):,}"
    )
    
    # CRITICAL FIX: Create separate dataset instances for each split
    # Previous bug: All subsets shared the same dataset, causing transform conflicts
    train_dataset_full = CheXmaskDataset(
        csv_path=config.CSV_PATH,
        image_base_path=config.IMAGE_BASE_PATH,
        dataset_name=config.DATASET_NAME,
        quality_threshold=config.QUALITY_THRESHOLD,
        transform=train_transform,  # Set transform directly
        mode='train',
        dataframe=df,
        path_lookup=path_lookup,
    )
    
    val_dataset_full = CheXmaskDataset(
        csv_path=config.CSV_PATH,
        image_base_path=config.IMAGE_BASE_PATH,
        dataset_name=config.DATASET_NAME,
        quality_threshold=config.QUALITY_THRESHOLD,
        transform=val_transform,  # Separate transform for validation
        mode='val',
        dataframe=df,
        path_lookup=path_lookup,
    )
    
    test_dataset_full = CheXmaskDataset(
        csv_path=config.CSV_PATH,
        image_base_path=config.IMAGE_BASE_PATH,
        dataset_name=config.DATASET_NAME,
        quality_threshold=config.QUALITY_THRESHOLD,
        transform=val_transform,  # No augmentation for test
        mode='test',
        dataframe=df,
        path_lookup=path_lookup,
    )
    
    # Create subset datasets with proper transforms
    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)
    test_dataset = torch.utils.data.Subset(test_dataset_full, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader


def _build_dataset_namespace(config, dataset_name: str, csv_path: str) -> SimpleNamespace:
    return SimpleNamespace(
        DATASET_NAME=dataset_name,
        CSV_PATH=csv_path,
        QUALITY_THRESHOLD=config.QUALITY_THRESHOLD,
        SPLIT_CONFIG=getattr(config, 'SPLIT_CONFIG', {}),
        VAL_RATIO=getattr(config, 'VAL_RATIO', 0.1),
        TRAIN_RATIO=getattr(config, 'TRAIN_RATIO', 0.7),
        TEST_RATIO=getattr(config, 'TEST_RATIO', 0.15),
        SEED=getattr(config, 'SEED', 42),
    )


def _get_composite_dataloaders(config, train_transform, val_transform):
    datasets = getattr(config, 'COMPOSITE_DATASETS', [])
    if not datasets:
        raise ValueError("USE_COMPOSITE_DATASET is True but COMPOSITE_DATASETS is empty.")

    train_parts: List[torch.utils.data.Dataset] = []
    val_parts: List[torch.utils.data.Dataset] = []
    test_parts: List[torch.utils.data.Dataset] = []
    summary: List[Tuple[str, int, int, int]] = []

    for dataset_name in datasets:
        csv_path = os.path.join(config.DATASET_DIR, "Preprocessed", f"{dataset_name}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV for dataset '{dataset_name}' not found at {csv_path}")

        df = load_dataset_dataframe(csv_path, config.QUALITY_THRESHOLD)
        dataset_cfg = _build_dataset_namespace(config, dataset_name, csv_path)
        id_col = detect_image_id_column(df)
        train_idx, val_idx, test_idx, extras = resolve_dataset_splits(dataset_cfg, df, id_col)
        path_lookup = extras.get('path_lookup', {}) if extras else {}

        image_base_path = config.IMAGE_ROOTS.get(dataset_name)
        if image_base_path is None:
            raise KeyError(f"IMAGE_ROOTS does not define a path for dataset '{dataset_name}'.")

        train_dataset_full = CheXmaskDataset(
            csv_path=csv_path,
            image_base_path=image_base_path,
            dataset_name=dataset_name,
            quality_threshold=config.QUALITY_THRESHOLD,
            transform=train_transform,
            mode='train',
            dataframe=df.copy(),
            path_lookup=path_lookup,
        )

        val_dataset_full = CheXmaskDataset(
            csv_path=csv_path,
            image_base_path=image_base_path,
            dataset_name=dataset_name,
            quality_threshold=config.QUALITY_THRESHOLD,
            transform=val_transform,
            mode='val',
            dataframe=df.copy(),
            path_lookup=path_lookup,
        )

        test_dataset_full = CheXmaskDataset(
            csv_path=csv_path,
            image_base_path=image_base_path,
            dataset_name=dataset_name,
            quality_threshold=config.QUALITY_THRESHOLD,
            transform=val_transform,
            mode='test',
            dataframe=df.copy(),
            path_lookup=path_lookup,
        )

        train_parts.append(torch.utils.data.Subset(train_dataset_full, train_idx))
        val_parts.append(torch.utils.data.Subset(val_dataset_full, val_idx))
        test_parts.append(torch.utils.data.Subset(test_dataset_full, test_idx))

        summary.append((dataset_name, len(train_idx), len(val_idx), len(test_idx)))

    train_dataset = ConcatDataset(train_parts)
    val_dataset = ConcatDataset(val_parts)
    test_dataset = ConcatDataset(test_parts)

    print("Composite dataset summary:")
    for name, train_count, val_count, test_count in summary:
        print(
            f"  {name:>12} -> train: {train_count:,} | val: {val_count:,} | "
            f"test: {test_count:,}"
        )
    print(
        "  Combined -> train: "
        f"{len(train_dataset):,} | val: {len(val_dataset):,} | "
        f"test: {len(test_dataset):,}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset loader
    import sys
    sys.path.append('..')
    import config

    df = load_dataset_dataframe(config.CSV_PATH, config.QUALITY_THRESHOLD)
    id_col = detect_image_id_column(df)
    _, _, _, extras = resolve_dataset_splits(config, df, id_col)
    path_lookup = extras.get('path_lookup', {}) if extras else {}
    
    # Create dataset
    dataset = CheXmaskDataset(
        csv_path=config.CSV_PATH,
        image_base_path=config.IMAGE_BASE_PATH,
        dataset_name=config.DATASET_NAME,
        quality_threshold=config.QUALITY_THRESHOLD,
        transform=None,
        mode='train',
        dataframe=df,
        path_lookup=path_lookup,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Mask shape: {sample['mask'].shape}")
        print(f"Unique mask values: {torch.unique(sample['mask'])}")
