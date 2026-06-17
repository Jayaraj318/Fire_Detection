"""
Dataset Download Scripts
Download fire detection datasets - Multiple methods available
"""

import os
import zipfile
import shutil
import urllib.request
from pathlib import Path
import argparse


def setup_kaggle_credentials(username: str, api_key: str):
    """
    Create kaggle.json file with your credentials.
    
    Args:
        username: Your Kaggle username
        api_key: Your Kaggle API token (starts with KGAT_...)
    """
    import json
    
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    credentials = {
        "username": username,
        "key": api_key
    }
    
    with open(kaggle_json, 'w') as f:
        json.dump(credentials, f)
    
    # Set permissions (important on Linux/Mac)
    try:
        os.chmod(kaggle_json, 0o600)
    except:
        pass
    
    print(f"✅ Kaggle credentials saved to: {kaggle_json}")
    print("   You can now use --kaggle option to download datasets")


def download_from_url(url: str, output_path: Path, desc: str = "file"):
    """Download a file from URL with progress."""
    print(f"📥 Downloading {desc}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"   ✅ Downloaded to: {output_path}")
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


def download_fire_datasets_direct():
    """
    Download fire detection datasets using direct URLs.
    No Kaggle API required!
    """
    output_dir = Path('data/raw')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔥 Downloading Fire Detection Datasets (Direct URLs)")
    print("=" * 50)
    
    # Alternative sources for fire datasets
    datasets = [
        {
            'name': 'Fire Dataset (GitHub Mirror)',
            'url': 'https://github.com/cair/Fire-Detection-Image-Dataset/archive/refs/heads/master.zip',
            'filename': 'fire_dataset.zip'
        },
    ]
    
    for dataset in datasets:
        output_path = output_dir / dataset['filename']
        success = download_from_url(dataset['url'], output_path, dataset['name'])
        
        if success and dataset['filename'].endswith('.zip'):
            print(f"   📦 Extracting {dataset['filename']}...")
            try:
                with zipfile.ZipFile(output_path, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
                print(f"   ✅ Extracted successfully")
                output_path.unlink()  # Remove zip after extraction
            except Exception as e:
                print(f"   ⚠️ Extraction warning: {e}")
    
    print("\n✅ Download complete!")
    print(f"   Files saved to: {output_dir.absolute()}")
    print("\n📋 Next steps:")
    print("1. Organize images into data/raw/images/")
    print("2. Create YOLO format annotations in data/raw/labels/")
    print("3. Run: python -m src.preprocessing.split_data")
    print("\n💡 Or use --sample to create test images quickly")


def download_kaggle_datasets():
    """
    Download fire detection datasets from Kaggle.
    
    Requires kaggle.json to be set up first.
    Run: python download_data.py --setup-kaggle YOUR_USERNAME YOUR_API_KEY
    """
    try:
        import kaggle
    except ImportError:
        print("Please install kaggle: pip install kaggle")
        print("\nThen set up credentials:")
        print("  python -m src.preprocessing.download_data --setup-kaggle YOUR_USERNAME YOUR_API_KEY")
        return
    
    output_dir = Path('data/raw')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # List of fire detection datasets on Kaggle
    datasets = [
        'phylake1337/fire-dataset',  # Basic fire dataset
        'naman29/d-fire-dataset',    # D-Fire: High quality, widely used
        'dataclusterlabs/fire-smoke-dataset', # Diverse scenes
        'ritupande/fire-detection-from-images',
    ]
    
    print("🔥 Downloading Fire Detection Datasets from Kaggle")
    print("=" * 50)
    
    for dataset in datasets:
        print(f"\n📥 Downloading: {dataset}")
        try:
            # Create a subfolder for each dataset to avoid conflicts
            dataset_name = dataset.split('/')[-1]
            dataset_path = output_dir / dataset_name
            dataset_path.mkdir(exist_ok=True)
            
            kaggle.api.dataset_download_files(
                dataset,
                path=str(dataset_path),
                unzip=True
            )
            print(f"   ✅ Downloaded into: {dataset_path}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            
    print("\n📦 Organizing datasets into standard structure...")
    organize_datasets(output_dir)
    
    print("\n✅ Download complete!")
    print(f"   Files saved to: {output_dir.absolute()}")
    print("\nNext steps:")
    print("1. Organize images into data/raw/images/")
    print("2. Create YOLO format annotations in data/raw/labels/")
    print("3. Run: python -m src.preprocessing.split_data")


def download_sample_images():
    """
    Create sample placeholder images for testing.
    """
    import numpy as np
    
    try:
        import cv2
    except ImportError:
        print("Please install opencv: pip install opencv-python")
        return
    
    output_dir = Path('data/raw/images')
    labels_dir = Path('data/raw/labels')
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating sample images for testing...")
    
    # Create sample fire images (red/orange)
    for i in range(5):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Add fire-like colors
        cv2.circle(img, (320, 320), 150, (0, 100, 255), -1)  # Orange
        cv2.circle(img, (320, 280), 100, (0, 0, 255), -1)  # Red
        cv2.circle(img, (320, 260), 60, (0, 200, 255), -1)  # Yellow
        
        # Add some noise
        noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        
        cv2.imwrite(str(output_dir / f'fire_sample_{i+1}.jpg'), img)
        
        # Create label (fire class = 0, center of image)
        with open(labels_dir / f'fire_sample_{i+1}.txt', 'w') as f:
            f.write('0 0.5 0.5 0.4 0.4\n')  # class x_center y_center width height
    
    # Create sample non-fire images
    for i in range(5):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Add some neutral colors
        cv2.rectangle(img, (100, 100), (540, 540), (100, 150, 100), -1)  # Green
        
        noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        
        cv2.imwrite(str(output_dir / f'nofire_sample_{i+1}.jpg'), img)
        
        # Empty label (no fire)
        with open(labels_dir / f'nofire_sample_{i+1}.txt', 'w') as f:
            pass  # Empty file
    
    print(f"✅ Created 10 sample images in {output_dir}")
    print("   - 5 fire samples")
    print("   - 5 non-fire samples")


def organize_datasets(base_dir: Path):
    """
    Consolidate all images and labels from subfolders into
    data/raw/images and data/raw/labels
    """
    target_images = base_dir / 'images'
    target_labels = base_dir / 'labels'
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)
    
    print(f"   Scanning {base_dir} for images...")
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    count = 0
    
    # Walk through all directories
    for root, _, files in os.walk(base_dir):
        root_path = Path(root)
        
        # Skip the target directories themselves to avoid recursion loops
        if root_path == target_images or root_path == target_labels:
            continue
            
        for file in files:
            file_path = root_path / file
            suffix = file_path.suffix.lower()
            
            if suffix in image_extensions:
                # Found an image
                try:
                    # Create a unique name to prevent overwrites
                    # Use parent folder name as prefix
                    prefix = root_path.name
                    if prefix == 'images':
                        prefix = root_path.parent.name
                    
                    new_name = f"{prefix}_{file}"
                    
                    # Move image
                    shutil.move(str(file_path), str(target_images / new_name))
                    
                    # Handle Labels
                    # Strategy 1: Look for existing .txt label
                    label_name = file_path.stem + '.txt'
                    found_label = False
                    
                    # Check common locations
                    possible_label_locs = [
                        root_path / label_name,
                        root_path.parent / 'labels' / label_name,
                        root_path / 'labels' / label_name,
                         # Some datasets have 'labels/train' structure matching 'images/train'
                        root_path.parent.parent / 'labels' / root_path.name / label_name
                    ]

                    for label_loc in possible_label_locs:
                        if label_loc.exists():
                            shutil.move(str(label_loc), str(target_labels / (Path(new_name).stem + '.txt')))
                            found_label = True
                            break
                    
                    # Strategy 2: Generate label from folder name (if no explicit label found)
                    if not found_label:
                        parent_name = root_path.name.lower()
                        grandparent_name = root_path.parent.name.lower()
                        
                        generated_label_file = target_labels / (Path(new_name).stem + '.txt')
                        
                        # Check for indication of FIRE
                        is_fire = 'fire' in parent_name and 'non' not in parent_name
                        is_fire = is_fire or ('fire' in grandparent_name and 'non' not in grandparent_name)
                        
                        # Explicit exclusion for 'neutral' or 'nofire'
                        if 'non' in parent_name or 'neutral' in parent_name or 'nofire' in parent_name:
                            is_fire = False

                        if is_fire:
                            # Create a dummy fire label (class 0, center box)
                            # We don't have bbox, but for classification or weak detection this might be useful
                            # Or we might skip adding a label file if we only want strong supervision?
                            # For now, let's create a full-image box 0 0.5 0.5 1 1 ? 
                            # Better: Don't create a label if we don't have bbox, UNLESS we are training a classifier.
                            # BUT, our training script treats MISSING label as NON-FIRE.
                            # So we MUST create a label for fire images.
                            with open(generated_label_file, 'w') as f:
                                # Create a generic "whole image is fire" box? 
                                # This is bad for detection training (IoU will be low).
                                # But better than "non-fire".
                                # Let's mark it as a TODO or warning.
                                # Actually, for datasets like phylake1337, they are classification datasets.
                                # Using them for detection is tricky without bboxes.
                                # maybe skip them or use a classification head?
                                # For now, let's just make it a small box in the center? No.
                                # Create a dummy fire label (class 0, center box, full size)
                                # This ensures the image is treated as FIRE even if detection is weak.
                                # Prevents treating fire images as background.
                                with open(generated_label_file, 'w') as f:
                                    f.write('0 0.5 0.5 0.99 0.99\n') 
                        else:
                            # Non-fire: create empty label file (confirmed background)
                            with open(generated_label_file, 'w') as f:
                                pass # Empty file = non-fire

                    count += 1
                except Exception as e:
                    print(f"   ⚠️ Error moving {file}: {e}")
                    
    print(f"   ✅ Moved {count} images and labels to standard folders")
    
    # Clean up empty folders (optional, but keeps things tidy)
    # for root, dirs, files in os.walk(base_dir, topdown=False):
    #     for name in dirs:
    #         try:
    #             (Path(root) / name).rmdir()
    #         except:
    #             pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download fire detection datasets')
    parser.add_argument('--kaggle', action='store_true', help='Download from Kaggle (requires credentials)')
    parser.add_argument('--direct', action='store_true', help='Download from direct URLs (no credentials needed)')
    parser.add_argument('--sample', action='store_true', help='Create sample test images')
    parser.add_argument('--setup-kaggle', nargs=2, metavar=('USERNAME', 'API_KEY'),
                       help='Set up Kaggle credentials')
    
    args = parser.parse_args()
    
    if args.setup_kaggle:
        setup_kaggle_credentials(args.setup_kaggle[0], args.setup_kaggle[1])
    elif args.kaggle:
        download_kaggle_datasets()
    elif args.direct:
        download_fire_datasets_direct()
    elif args.sample:
        download_sample_images()
    else:
        print("🔥 Fire Detection Dataset Downloader")
        print("=" * 50)
        print("\nUsage:")
        print("  python -m src.preprocessing.download_data --direct   # Download from GitHub (recommended)")
        print("  python -m src.preprocessing.download_data --sample   # Create sample test images")
        print("  python -m src.preprocessing.download_data --kaggle   # Download from Kaggle")
        print("\nKaggle Setup:")
        print("  python -m src.preprocessing.download_data --setup-kaggle YOUR_USERNAME YOUR_API_KEY")

