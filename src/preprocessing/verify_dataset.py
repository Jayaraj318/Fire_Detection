"""
Dataset Verification Utilities
Analyze and verify fire detection dataset quality.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from collections import Counter
import json


def verify_annotations(data_dir: str, verbose: bool = True) -> Dict:
    """
    Verify YOLO annotation format and validity.
    
    Args:
        data_dir: Path to dataset split (e.g., data/splits/train)
        verbose: Print detailed info
        
    Returns:
        Dict with verification results
    """
    data_path = Path(data_dir)
    images_dir = data_path / 'images'
    labels_dir = data_path / 'labels'
    
    results = {
        'valid': True,
        'total_images': 0,
        'total_labels': 0,
        'images_without_labels': [],
        'labels_without_images': [],
        'invalid_annotations': [],
        'annotation_issues': [],
    }
    
    if not images_dir.exists():
        results['valid'] = False
        results['annotation_issues'].append(f"Images directory not found: {images_dir}")
        return results
    
    if not labels_dir.exists():
        results['valid'] = False
        results['annotation_issues'].append(f"Labels directory not found: {labels_dir}")
        return results
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images = {f.stem: f for f in images_dir.iterdir() 
              if f.suffix.lower() in image_extensions}
    
    # Find all labels
    labels = {f.stem: f for f in labels_dir.iterdir() if f.suffix == '.txt'}
    
    results['total_images'] = len(images)
    results['total_labels'] = len(labels)
    
    # Check for missing pairs
    for img_name in images:
        if img_name not in labels:
            results['images_without_labels'].append(img_name)
    
    for label_name in labels:
        if label_name not in images:
            results['labels_without_images'].append(label_name)
    
    # Validate annotation format
    for label_name, label_path in labels.items():
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                
                # YOLO format: class x_center y_center width height
                if len(parts) < 5:
                    results['invalid_annotations'].append({
                        'file': label_name,
                        'line': line_num,
                        'issue': f"Expected 5 values, got {len(parts)}"
                    })
                    continue
                
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Check normalized coordinates are in [0, 1]
                    if not (0 <= x_center <= 1):
                        results['annotation_issues'].append({
                            'file': label_name,
                            'line': line_num,
                            'issue': f"x_center out of range: {x_center}"
                        })
                    if not (0 <= y_center <= 1):
                        results['annotation_issues'].append({
                            'file': label_name,
                            'line': line_num,
                            'issue': f"y_center out of range: {y_center}"
                        })
                    if not (0 < width <= 1):
                        results['annotation_issues'].append({
                            'file': label_name,
                            'line': line_num,
                            'issue': f"width out of range: {width}"
                        })
                    if not (0 < height <= 1):
                        results['annotation_issues'].append({
                            'file': label_name,
                            'line': line_num,
                            'issue': f"height out of range: {height}"
                        })
                    
                    # Check for placeholder annotations (centered box)
                    if (abs(x_center - 0.5) < 0.01 and 
                        abs(y_center - 0.5) < 0.01 and
                        abs(width - 0.8) < 0.01 and 
                        abs(height - 0.8) < 0.01):
                        results['annotation_issues'].append({
                            'file': label_name,
                            'line': line_num,
                            'issue': "Placeholder annotation detected (0.5 0.5 0.8 0.8)"
                        })
                        
                except ValueError as e:
                    results['invalid_annotations'].append({
                        'file': label_name,
                        'line': line_num,
                        'issue': f"Invalid number format: {e}"
                    })
                    
        except Exception as e:
            results['invalid_annotations'].append({
                'file': label_name,
                'issue': f"Failed to read: {e}"
            })
    
    # Set validity
    if results['invalid_annotations'] or len(results['images_without_labels']) > 0.1 * len(images):
        results['valid'] = False
    
    if verbose:
        print(f"\n📊 Dataset Verification: {data_dir}")
        print("=" * 50)
        print(f"   Total images: {results['total_images']}")
        print(f"   Total labels: {results['total_labels']}")
        print(f"   Images without labels: {len(results['images_without_labels'])}")
        print(f"   Labels without images: {len(results['labels_without_images'])}")
        print(f"   Invalid annotations: {len(results['invalid_annotations'])}")
        print(f"   Annotation issues: {len(results['annotation_issues'])}")
        print(f"   Valid: {'✅' if results['valid'] else '❌'}")
        
        if results['annotation_issues'][:5]:
            print(f"\n⚠️ Sample issues:")
            for issue in results['annotation_issues'][:5]:
                print(f"   - {issue['file']}: {issue['issue']}")
    
    return results


def analyze_class_distribution(data_dir: str, verbose: bool = True) -> Dict:
    """
    Analyze fire/non-fire class distribution.
    
    Args:
        data_dir: Path to dataset split
        verbose: Print detailed info
        
    Returns:
        Dict with class distribution statistics
    """
    data_path = Path(data_dir)
    labels_dir = data_path / 'labels'
    
    results = {
        'fire_samples': 0,
        'non_fire_samples': 0,
        'total_samples': 0,
        'fire_ratio': 0.0,
        'non_fire_ratio': 0.0,
        'is_balanced': False,
        'class_counts': Counter(),
    }
    
    if not labels_dir.exists():
        return results
    
    for label_file in labels_dir.glob('*.txt'):
        with open(label_file, 'r') as f:
            content = f.read().strip()
        
        results['total_samples'] += 1
        
        if content:
            # Has annotations - check for fire class
            has_fire = False
            for line in content.split('\n'):
                parts = line.strip().split()
                if len(parts) >= 1:
                    class_id = int(parts[0])
                    results['class_counts'][class_id] += 1
                    if class_id == 0:  # Fire class
                        has_fire = True
            
            if has_fire:
                results['fire_samples'] += 1
            else:
                results['non_fire_samples'] += 1
        else:
            # Empty label = non-fire
            results['non_fire_samples'] += 1
    
    if results['total_samples'] > 0:
        results['fire_ratio'] = results['fire_samples'] / results['total_samples']
        results['non_fire_ratio'] = results['non_fire_samples'] / results['total_samples']
        
        # Consider balanced if ratio is between 40-60%
        results['is_balanced'] = 0.4 <= results['fire_ratio'] <= 0.6
    
    if verbose:
        print(f"\n📈 Class Distribution: {data_dir}")
        print("=" * 50)
        print(f"   🔥 Fire samples:     {results['fire_samples']:>5} ({results['fire_ratio']*100:.1f}%)")
        print(f"   ✅ Non-fire samples: {results['non_fire_samples']:>5} ({results['non_fire_ratio']*100:.1f}%)")
        print(f"   📊 Total samples:    {results['total_samples']:>5}")
        print(f"   ⚖️ Balanced: {'✅' if results['is_balanced'] else '❌ (aim for 40-60% fire)'}")
        
        if results['class_counts']:
            print(f"\n   Class breakdown:")
            for class_id, count in sorted(results['class_counts'].items()):
                print(f"     Class {class_id}: {count}")
    
    return results


def compute_weighted_sampler_weights(data_dir: str) -> Tuple[List[float], Dict]:
    """
    Compute sample weights for PyTorch WeightedRandomSampler.
    
    Args:
        data_dir: Path to dataset split
        
    Returns:
        Tuple of (weight_list, info_dict)
    """
    data_path = Path(data_dir)
    labels_dir = data_path / 'labels'
    images_dir = data_path / 'images'
    
    # Get all image files in order
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = sorted([f for f in images_dir.iterdir() 
                          if f.suffix.lower() in image_extensions])
    
    # Count classes
    fire_count = 0
    non_fire_count = 0
    sample_classes = []  # 1 for fire, 0 for non-fire
    
    for img_file in image_files:
        label_file = labels_dir / f"{img_file.stem}.txt"
        
        is_fire = False
        if label_file.exists():
            with open(label_file, 'r') as f:
                content = f.read().strip()
            
            if content:
                for line in content.split('\n'):
                    parts = line.strip().split()
                    if len(parts) >= 1 and int(parts[0]) == 0:
                        is_fire = True
                        break
        
        if is_fire:
            fire_count += 1
            sample_classes.append(1)
        else:
            non_fire_count += 1
            sample_classes.append(0)
    
    # Compute weights (inverse frequency)
    total = fire_count + non_fire_count
    if fire_count > 0 and non_fire_count > 0:
        fire_weight = total / (2 * fire_count)
        non_fire_weight = total / (2 * non_fire_count)
    else:
        fire_weight = 1.0
        non_fire_weight = 1.0
    
    # Assign weights to each sample
    weights = [fire_weight if c == 1 else non_fire_weight for c in sample_classes]
    
    info = {
        'fire_count': fire_count,
        'non_fire_count': non_fire_count,
        'fire_weight': fire_weight,
        'non_fire_weight': non_fire_weight,
    }
    
    print(f"\n⚖️ Weighted Sampling Info:")
    print(f"   Fire samples: {fire_count} (weight: {fire_weight:.3f})")
    print(f"   Non-fire samples: {non_fire_count} (weight: {non_fire_weight:.3f})")
    
    return weights, info


def verify_dataset(data_dir: str = 'data/splits', verbose: bool = True) -> Dict:
    """
    Run full dataset verification.
    
    Args:
        data_dir: Path to data splits directory
        verbose: Print detailed info
        
    Returns:
        Combined verification results
    """
    data_path = Path(data_dir)
    results = {}
    
    print("\n" + "=" * 60)
    print("🔍 FIRE DETECTION DATASET VERIFICATION")
    print("=" * 60)
    
    for split in ['train', 'val', 'test']:
        split_dir = data_path / split
        if split_dir.exists():
            print(f"\n{'─' * 40}")
            print(f"📁 Split: {split.upper()}")
            print(f"{'─' * 40}")
            
            results[split] = {
                'annotations': verify_annotations(str(split_dir), verbose),
                'distribution': analyze_class_distribution(str(split_dir), verbose),
            }
        else:
            if verbose:
                print(f"\n⚠️ Split '{split}' not found at {split_dir}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    
    all_valid = True
    total_fire = 0
    total_non_fire = 0
    
    for split, data in results.items():
        valid = data['annotations']['valid']
        all_valid = all_valid and valid
        total_fire += data['distribution']['fire_samples']
        total_non_fire += data['distribution']['non_fire_samples']
        
        status = "✅ Valid" if valid else "❌ Issues found"
        print(f"   {split:>5}: {status}")
    
    total = total_fire + total_non_fire
    if total > 0:
        print(f"\n   Overall balance: {total_fire}/{total} fire ({total_fire/total*100:.1f}%)")
        
        if total < 1000:
            print(f"\n   ⚠️ WARNING: Only {total} total samples. Recommend 1000+ per class.")
        
        if total_fire / total < 0.4 or total_fire / total > 0.6:
            print(f"   ⚠️ WARNING: Dataset is imbalanced. Consider weighted sampling.")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify fire detection dataset')
    parser.add_argument('--data', type=str, default='data/splits', 
                       help='Path to data splits directory')
    parser.add_argument('--split', type=str, default=None,
                       help='Specific split to verify (train/val/test)')
    parser.add_argument('--json', action='store_true',
                       help='Output results as JSON')
    
    args = parser.parse_args()
    
    if args.split:
        split_dir = Path(args.data) / args.split
        results = {
            'annotations': verify_annotations(str(split_dir)),
            'distribution': analyze_class_distribution(str(split_dir)),
        }
    else:
        results = verify_dataset(args.data)
    
    if args.json:
        # Convert to JSON-serializable format
        def convert(obj):
            if isinstance(obj, Counter):
                return dict(obj)
            if isinstance(obj, Path):
                return str(obj)
            return obj
        
        import json
        print(json.dumps(results, default=convert, indent=2))
