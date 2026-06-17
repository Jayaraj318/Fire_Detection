"""
Data Preprocessing and Splitting
Prepare fire detection dataset for training
"""

import os
import shutil
from pathlib import Path
import random
from sklearn.model_selection import train_test_split
import argparse


def split_dataset(
    source_dir: str = 'data/raw',
    output_dir: str = 'data/splits',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    """
    Split dataset into train/val/test sets.
    
    Expected source structure:
    data/raw/
    ├── images/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── labels/
        ├── image1.txt
        ├── image2.txt
        └── ...
    
    Output structure:
    data/splits/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
    """
    random.seed(seed)
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Find all images
    images_dir = source_path / 'images'
    labels_dir = source_path / 'labels'
    
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        print("Please create the directory and add your images.")
        return
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    images = []
    for ext in image_extensions:
        images.extend(list(images_dir.glob(f'*{ext}')))
        images.extend(list(images_dir.glob(f'*{ext.upper()}')))
    
    if len(images) == 0:
        print(f"No images found in {images_dir}")
        print("Supported formats: JPG, PNG, BMP, WebP")
        return
    
    print(f"Found {len(images)} images")
    
    # Split dataset
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "Ratios must sum to 1.0"
    
    train_imgs, temp_imgs = train_test_split(
        images, 
        test_size=(val_ratio + test_ratio), 
        random_state=seed
    )
    
    # Split remaining into val and test
    relative_val = val_ratio / (val_ratio + test_ratio)
    val_imgs, test_imgs = train_test_split(
        temp_imgs, 
        test_size=(1 - relative_val), 
        random_state=seed
    )
    
    print(f"Split sizes:")
    print(f"  Train: {len(train_imgs)} ({train_ratio:.0%})")
    print(f"  Val:   {len(val_imgs)} ({val_ratio:.0%})")
    print(f"  Test:  {len(test_imgs)} ({test_ratio:.0%})")
    
    # Create output directories and copy files
    for split_name, split_images in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
        split_dir = output_path / split_name
        images_out = split_dir / 'images'
        labels_out = split_dir / 'labels'
        
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
        
        for img_path in split_images:
            # Copy image
            shutil.copy(img_path, images_out / img_path.name)
            
            # Copy label if exists
            label_path = labels_dir / f'{img_path.stem}.txt'
            if label_path.exists():
                shutil.copy(label_path, labels_out / f'{img_path.stem}.txt')
    
    print(f"\n✅ Dataset split complete!")
    print(f"   Output: {output_path}")


def create_empty_labels(images_dir: str):
    """
    Create empty label files for images without labels.
    Useful for non-fire (negative) samples.
    """
    images_path = Path(images_dir)
    labels_path = images_path.parent / 'labels'
    labels_path.mkdir(exist_ok=True)
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    
    count = 0
    for ext in image_extensions:
        for img in images_path.glob(f'*{ext}'):
            label_file = labels_path / f'{img.stem}.txt'
            if not label_file.exists():
                label_file.touch()
                count += 1
    
    print(f"Created {count} empty label files")


def verify_dataset(dataset_dir: str):
    """
    Verify dataset integrity and print statistics.
    """
    dataset_path = Path(dataset_dir)
    
    for split in ['train', 'val', 'test']:
        split_dir = dataset_path / split
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        if not split_dir.exists():
            print(f"⚠️ {split} split not found")
            continue
        
        # Count images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        images = []
        for ext in image_extensions:
            images.extend(list(images_dir.glob(f'*{ext}')))
            images.extend(list(images_dir.glob(f'*{ext.upper()}')))
        
        # Count labels
        labels = list(labels_dir.glob('*.txt')) if labels_dir.exists() else []
        
        # Count fire and non-fire
        fire_count = 0
        for label_file in labels:
            with open(label_file, 'r') as f:
                content = f.read().strip()
                if content:  # Has annotations
                    fire_count += 1
        
        print(f"\n{split.upper()} Split:")
        print(f"  Images: {len(images)}")
        print(f"  Labels: {len(labels)}")
        print(f"  With fire: {fire_count}")
        print(f"  Without fire: {len(images) - fire_count}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare fire detection dataset')
    parser.add_argument('--source', type=str, default='data/raw', help='Source directory')
    parser.add_argument('--output', type=str, default='data/splits', help='Output directory')
    parser.add_argument('--verify', action='store_true', help='Verify existing dataset')
    
    args = parser.parse_args()
    
    if args.verify:
        verify_dataset(args.output)
    else:
        split_dataset(args.source, args.output)
        verify_dataset(args.output)
