"""
Extract RAR files and organize the fire detection dataset
"""

import rarfile
import os
import shutil
from pathlib import Path

# Set paths
raw_dir = Path('data/raw')
dataset_dir = raw_dir / 'Fire-Detection-Image-Dataset-master'
images_dir = raw_dir / 'images'
labels_dir = raw_dir / 'labels'

# Create output directories
images_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)

# RAR files to extract
rar_files = [
    'Fire images.rar',
    'Normal Images 1.rar',
    'Normal Images 2.rar', 
    'Normal Images 3.rar',
    'Normal Images 4.rar',
    'Normal Images 5.rar'
]

print("🔥 Extracting Fire Detection Dataset")
print("=" * 50)

for rar_file in rar_files:
    rar_path = dataset_dir / rar_file
    if rar_path.exists():
        print(f"\n📦 Extracting: {rar_file}")
        try:
            rf = rarfile.RarFile(str(rar_path))
            rf.extractall(str(raw_dir))
            print(f"   ✅ Extracted successfully")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    else:
        print(f"   ⚠️ Not found: {rar_file}")

# Now organize the extracted images
print("\n📁 Organizing images...")

# Find all extracted image directories and move images
fire_count = 0
normal_count = 0

for folder in raw_dir.iterdir():
    if folder.is_dir() and folder.name != 'images' and folder.name != 'labels':
        # Look for images in this folder
        for img_file in folder.rglob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                # Determine if fire or normal
                is_fire = 'fire' in str(img_file).lower() and 'non' not in str(img_file).lower()
                
                # Create unique filename
                if is_fire:
                    prefix = 'fire'
                    fire_count += 1
                    count = fire_count
                else:
                    prefix = 'normal'
                    normal_count += 1
                    count = normal_count
                
                new_name = f"{prefix}_{count:04d}{img_file.suffix.lower()}"
                dest_path = images_dir / new_name
                
                try:
                    shutil.copy2(img_file, dest_path)
                    
                    # Create label file
                    label_path = labels_dir / f"{prefix}_{count:04d}.txt"
                    with open(label_path, 'w') as f:
                        if is_fire:
                            # Fire class = 0, centered bounding box (will be refined with annotation)
                            f.write("0 0.5 0.5 0.8 0.8\n")
                        # Empty label for normal images (no fire)
                except Exception as e:
                    print(f"   ⚠️ Error copying {img_file.name}: {e}")

print(f"\n✅ Dataset organization complete!")
print(f"   🔥 Fire images: {fire_count}")
print(f"   ✅ Normal images: {normal_count}")
print(f"   📂 Total: {fire_count + normal_count}")
print(f"\nImages saved to: {images_dir.absolute()}")
print(f"Labels saved to: {labels_dir.absolute()}")
print("\n📋 Next step: Run 'python -m src.preprocessing.split_data'")
