"""
Inference script for Swin-UNet segmentation model
"""

import os
import sys
import argparse
from types import SimpleNamespace
from typing import List, Tuple

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models.swin_unet_pretrained import build_pretrained_swin_unet
from utils.dataset import (
    CheXmaskDataset,
    detect_image_id_column,
    load_dataset_dataframe,
    resolve_dataset_splits,
)
from utils.losses import SegmentationMetrics
from utils.visualization import visualize_segmentation, create_comparison_grid, save_prediction_as_mask


class Predictor:
    """Predictor class for segmentation model"""
    
    def __init__(self, config, checkpoint_path):
        self.config = config
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Build model
        print("Building model...")
        self.model = build_pretrained_swin_unet(config)
        self.model = self.model.to(self.device)
        
        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False  # allow loading checkpoints saved with Pickle globals
        )
        
        # Handle DataParallel models
        state_dict = checkpoint['model_state_dict']
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Drop attention masks since they are shape-dependent buffers rebuilt at runtime
        attn_mask_keys = [k for k in state_dict.keys() if 'attn_mask' in k]
        for key in attn_mask_keys:
            state_dict.pop(key)

        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        
        print(f"Loaded model from epoch {checkpoint['epoch']+1}")
        print(f"Best validation Dice: {checkpoint.get('best_val_dice', 'N/A')}")
    
    def predict_image(self, image):
        """
        Predict segmentation mask for a single image
        
        Args:
            image: (H, W) or (1, H, W) - grayscale image
            
        Returns:
            pred_mask: (H, W) - predicted class mask
            prob_map: (C, H, W) - probability map for each class
        """
        # Prepare image
        if len(image.shape) == 2:
            image = image[np.newaxis, :]
        
        # Normalize
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        # Resize to model input size
        orig_h, orig_w = image.shape[1:]
        image_resized = cv2.resize(image[0], (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE))
        image_resized = image_resized[np.newaxis, np.newaxis, :]
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_resized).float().to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(image_tensor)
            prob_map = F.softmax(output, dim=1)
            pred_mask = torch.argmax(output, dim=1)
        
        # Resize back to original size
        prob_map = F.interpolate(
            prob_map,
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        )
        pred_mask = F.interpolate(
            pred_mask.unsqueeze(1).float(),
            size=(orig_h, orig_w),
            mode='nearest'
        ).squeeze(1).long()
        
        # Convert to numpy
        pred_mask = pred_mask.cpu().numpy()[0]
        prob_map = prob_map.cpu().numpy()[0]
        
        return pred_mask, prob_map
    
    def predict_dataset(
        self,
        dataset,
        output_dir,
        save_masks=True,
        save_visualizations=True,
        dataset_label=None,
        max_samples=None,
    ):
        """
        Run predictions on entire dataset
        
        Args:
            dataset: CheXmaskDataset instance
            output_dir: directory to save results
            save_masks: whether to save prediction masks
            save_visualizations: whether to save visualizations
            dataset_label: optional label used for logging and result tracking
            max_samples: optional cap on number of samples to evaluate
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if save_masks:
            masks_dir = os.path.join(output_dir, 'masks')
            os.makedirs(masks_dir, exist_ok=True)
        
        if save_visualizations:
            vis_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
        
        # Initialize metrics
        metrics = SegmentationMetrics(self.config.NUM_CLASSES, self.device)
        
        # Results list
        results = []
        
        dataset_desc = dataset_label or 'dataset'
        total_samples = len(dataset)
        if max_samples is not None and max_samples < total_samples:
            print(
                f"Limiting inference for {dataset_desc} to {max_samples} of "
                f"{total_samples} samples"
            )
            total_samples = max_samples

        print(f"Running inference on {total_samples} samples ({dataset_desc})...")
        
        for idx in tqdm(range(total_samples)):
            # Get sample
            sample = dataset[idx]
            image = sample['image'].numpy()[0]  # Remove channel dimension
            gt_mask = sample['mask'].numpy()
            image_id = sample['image_id']
            sample_dataset = sample.get('dataset', dataset_label)
            if sample_dataset is None:
                sample_dataset = 'unknown'
            
            # Predict
            pred_mask, prob_map = self.predict_image(image)
            
            # Calculate metrics for this sample
            with torch.no_grad():
                pred_tensor = torch.from_numpy(pred_mask).unsqueeze(0).to(self.device)
                gt_tensor = torch.from_numpy(gt_mask).unsqueeze(0).to(self.device)
                
                # Create one-hot predictions for metrics
                pred_one_hot = F.one_hot(pred_tensor, self.config.NUM_CLASSES).permute(0, 3, 1, 2).float()
                
                metrics.update(pred_one_hot, gt_tensor)
            
            # Calculate per-sample metrics
            sample_metrics = {}
            for c in range(self.config.NUM_CLASSES):
                pred_c = (pred_mask == c)
                gt_c = (gt_mask == c)
                
                tp = np.logical_and(pred_c, gt_c).sum()
                fp = np.logical_and(pred_c, np.logical_not(gt_c)).sum()
                fn = np.logical_and(np.logical_not(pred_c), gt_c).sum()
                
                dice = 2 * tp / (2 * tp + fp + fn + 1e-7)
                iou = tp / (tp + fp + fn + 1e-7)
                
                sample_metrics[f'dice_class_{c}'] = dice
                sample_metrics[f'iou_class_{c}'] = iou
            
            # Store results
            results.append({
                'dataset': sample_dataset,
                'image_id': image_id,
                'mean_dice': np.mean([sample_metrics[f'dice_class_{c}'] for c in range(1, self.config.NUM_CLASSES)]),
                'mean_iou': np.mean([sample_metrics[f'iou_class_{c}'] for c in range(1, self.config.NUM_CLASSES)]),
                **sample_metrics
            })
            
            # Save mask
            if save_masks:
                mask_path = os.path.join(masks_dir, f'{image_id}_pred.png')
                save_prediction_as_mask(pred_mask, mask_path)
            
            # Save visualization
            if save_visualizations and idx < 50:  # Save first 50 samples
                vis_path = os.path.join(vis_dir, f'{image_id}_comparison.png')
                visualize_segmentation(image, gt_mask, pred_mask, save_path=vis_path)
                
                # Also save detailed comparison
                comp_path = os.path.join(vis_dir, f'{image_id}_detailed.png')
                create_comparison_grid(image, gt_mask, pred_mask, save_path=comp_path)
        
        # Compute overall metrics
        overall_metrics = metrics.get_mean_metrics()
        
        print("\n" + "="*50)
        print("Overall Metrics:")
        print(f"  Mean Dice: {overall_metrics['mean_dice']:.4f}")
        print(f"  Mean IoU: {overall_metrics['mean_iou']:.4f}")
        print("-"*50)
        
        class_names = ['Background', 'Right Lung', 'Left Lung', 'Heart']
        for c in range(self.config.NUM_CLASSES):
            print(f"  {class_names[c]}:")
            print(f"    Dice: {overall_metrics[f'dice_class_{c}']:.4f}")
            print(f"    IoU: {overall_metrics[f'iou_class_{c}']:.4f}")
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_path = os.path.join(output_dir, 'predictions_metrics.csv')
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to {results_path}")
        
        # Save overall metrics
        metrics_path = os.path.join(output_dir, 'overall_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write("Overall Metrics\n")
            f.write("="*50 + "\n")
            for key, value in overall_metrics.items():
                f.write(f"{key}: {value:.4f}\n")
        print(f"Overall metrics saved to {metrics_path}")
        
        return results_df, overall_metrics
    
    def predict_single_image(self, image_path, save_path=None):
        """
        Predict segmentation for a single image file
        
        Args:
            image_path: path to image file
            save_path: path to save visualization (optional)
        """
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Predict
        pred_mask, prob_map = self.predict_image(image)
        
        # Visualize
        if save_path:
            visualize_segmentation(image / 255.0, pred_mask, None, save_path=save_path)
        
        return pred_mask, prob_map


def build_eval_transform(cfg):
    """Create the evaluation transform (resize-only)."""
    import albumentations as A

    return A.Compose([
        A.Resize(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
    ])


def determine_target_datasets(args, cfg) -> List[str]:
    """Resolve which dataset(s) should be evaluated."""

    composite_datasets = list(getattr(cfg, 'COMPOSITE_DATASETS', []))
    available_roots = getattr(cfg, 'IMAGE_ROOTS', {})
    known_datasets = set(available_roots.keys()) | {cfg.DATASET_NAME}

    if args.dataset_name:
        requested = args.dataset_name.strip()
        if requested.lower() == 'all':
            if composite_datasets:
                return composite_datasets
            return [cfg.DATASET_NAME]
        if requested not in known_datasets:
            raise ValueError(
                f"Unknown dataset '{requested}'. Known datasets: {sorted(known_datasets)}"
            )
        return [requested]

    if args.composite or getattr(cfg, 'USE_COMPOSITE_DATASET', False):
        if not composite_datasets:
            raise ValueError(
                "Composite inference requested but COMPOSITE_DATASETS is empty in config."
            )
        return composite_datasets

    return [cfg.DATASET_NAME]


def _build_dataset_namespace(cfg, dataset_name: str, csv_path: str) -> SimpleNamespace:
    """Helper to mirror the fields expected by resolve_dataset_splits."""

    return SimpleNamespace(
        DATASET_NAME=dataset_name,
        CSV_PATH=csv_path,
        QUALITY_THRESHOLD=getattr(cfg, 'QUALITY_THRESHOLD', 0.0),
        SPLIT_CONFIG=getattr(cfg, 'SPLIT_CONFIG', {}),
        VAL_RATIO=getattr(cfg, 'VAL_RATIO', 0.1),
        TRAIN_RATIO=getattr(cfg, 'TRAIN_RATIO', 0.7),
        TEST_RATIO=getattr(cfg, 'TEST_RATIO', 0.15),
        SEED=getattr(cfg, 'SEED', 42),
    )


def create_dataset_for_split(
    cfg,
    dataset_name: str,
    split: str,
    transform,
) -> Tuple[torch.utils.data.Dataset, int]:
    """Instantiate a dataset subset for the requested split."""

    dataset_name = str(dataset_name)
    split = split.lower()

    if dataset_name == cfg.DATASET_NAME:
        csv_path = cfg.CSV_PATH
    else:
        dataset_dir = getattr(cfg, 'DATASET_DIR', None)
        if dataset_dir is None:
            raise ValueError("config is missing DATASET_DIR needed for composite datasets.")
        csv_path = os.path.join(dataset_dir, 'Preprocessed', f'{dataset_name}.csv')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV for dataset '{dataset_name}' not found at {csv_path}")

    dataframe = load_dataset_dataframe(csv_path, cfg.QUALITY_THRESHOLD, dataset_name)
    dataset_cfg = _build_dataset_namespace(cfg, dataset_name, csv_path)
    id_col = detect_image_id_column(dataframe)
    train_idx, val_idx, test_idx, extras = resolve_dataset_splits(dataset_cfg, dataframe, id_col)

    split_indices = {
        'train': train_idx,
        'val': val_idx,
        'test': test_idx,
    }.get(split)

    if split_indices is None:
        raise ValueError(f"Unsupported split '{split}'. Choose from 'train', 'val', or 'test'.")

    path_lookup = extras.get('path_lookup', {}) if extras else {}

    image_roots = getattr(cfg, 'IMAGE_ROOTS', {})
    image_base_path = image_roots.get(dataset_name)
    if image_base_path is None and dataset_name == cfg.DATASET_NAME:
        image_base_path = getattr(cfg, 'IMAGE_BASE_PATH', None)
    if image_base_path is None:
        raise KeyError(
            f"IMAGE_ROOTS does not define a path for dataset '{dataset_name}'. Update config."
        )

    dataset_full = CheXmaskDataset(
        csv_path=csv_path,
        image_base_path=image_base_path,
        dataset_name=dataset_name,
        quality_threshold=cfg.QUALITY_THRESHOLD,
        transform=transform,
        mode=split,
        dataframe=dataframe.copy(),
        path_lookup=path_lookup,
    )

    subset = torch.utils.data.Subset(dataset_full, split_indices)
    return subset, len(split_indices)


def prepare_datasets(cfg, dataset_names: List[str], split: str, transform):
    """Prepare the datasets to be evaluated for the selected split."""

    prepared = []
    for name in dataset_names:
        subset, count = create_dataset_for_split(cfg, name, split, transform)
        if count == 0:
            print(f"[Warning] Dataset '{name}' has no samples in split '{split}'. Skipping.")
            continue
        prepared.append((name, subset, count))
    return prepared


def main():
    parser = argparse.ArgumentParser(description='Run inference with Swin-UNet')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--split', '--dataset', dest='split', type=str, default='test',
                        choices=['train', 'val', 'test'], help='Which dataset split to use')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for results')
    parser.add_argument('--save_masks', action='store_true', help='Save prediction masks')
    parser.add_argument('--save_vis', action='store_true', help='Save visualizations')
    parser.add_argument('--image_path', type=str, default=None, help='Path to single image for prediction')
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='Specific dataset to evaluate (overrides config). Use "all" for composite.')
    parser.add_argument('--composite', action='store_true',
                        help='Force evaluation across all composite datasets defined in config.')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Optional cap on samples per dataset for quick smoke tests.')
    args = parser.parse_args()
    
    # Create predictor
    predictor = Predictor(config, args.checkpoint)
    
    # Single image prediction
    if args.image_path:
        output_path = args.output_dir or 'prediction.png'
        pred_mask, prob_map = predictor.predict_single_image(args.image_path, save_path=output_path)
        print(f"Prediction saved to {output_path}")
        print(f"Predicted classes: {np.unique(pred_mask)}")
        return
    
    split = args.split

    eval_transform = build_eval_transform(config)
    target_datasets = determine_target_datasets(args, config)
    prepared_datasets = prepare_datasets(config, target_datasets, split, eval_transform)

    if not prepared_datasets:
        raise ValueError("No datasets available for inference. Check dataset configuration and splits.")

    base_output_dir = args.output_dir or os.path.join(
        config.OUTPUT_DIR,
        config.EXPERIMENT_NAME,
        f'inference_{split}'
    )

    summary_rows = []

    for dataset_name, dataset_obj, original_count in prepared_datasets:
        print(
            f"\n=== Evaluating {dataset_name} split '{split}' "
            f"({original_count} samples available) ==="
        )

        dataset_output_dir = base_output_dir
        if len(prepared_datasets) > 1:
            dataset_output_dir = os.path.join(base_output_dir, dataset_name)

        results_df, overall_metrics = predictor.predict_dataset(
            dataset_obj,
            dataset_output_dir,
            save_masks=args.save_masks,
            save_visualizations=args.save_vis,
            dataset_label=dataset_name,
            max_samples=args.max_samples,
        )

        evaluated_samples = len(results_df)
        summary_rows.append((dataset_name, evaluated_samples, overall_metrics, dataset_output_dir))

        print(
            f"\nFinished {dataset_name}: evaluated {evaluated_samples} samples. "
            f"Outputs saved in {dataset_output_dir}"
        )

    if len(summary_rows) > 1:
        print("\nSummary across datasets:")
        for name, evaluated_samples, metrics, output_dir in summary_rows:
            mean_dice = metrics.get('mean_dice')
            mean_iou = metrics.get('mean_iou')
            dice_str = f"{mean_dice:.4f}" if isinstance(mean_dice, (int, float)) else str(mean_dice)
            iou_str = f"{mean_iou:.4f}" if isinstance(mean_iou, (int, float)) else str(mean_iou)
            print(
                f"  {name}: samples={evaluated_samples} | "
                f"mean_dice={dice_str} | mean_iou={iou_str} | outputs={output_dir}"
            )


if __name__ == '__main__':
    main()
