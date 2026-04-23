import os, shutil
from sklearn.model_selection import train_test_split

# Paths
base_dir = "dataset thyroid"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")

# Ensure validation dirs exist
for category in ["benign", "malignant"]:
    os.makedirs(os.path.join(val_dir, category), exist_ok=True)

# Split train into train+validation
for category in ["benign", "malignant"]:
    src_folder = os.path.join(train_dir, category)
    imgs = os.listdir(src_folder)

    # Filter only image files (ignore hidden/system files)
    imgs = [img for img in imgs if img.lower().endswith((".png", ".jpg", ".jpeg"))]

    # Skip if already split (avoid moving twice)
    imgs_to_move = [img for img in imgs if not os.path.exists(os.path.join(val_dir, category, img))]

    if len(imgs_to_move) > 0:
        train_imgs, val_imgs = train_test_split(imgs_to_move, test_size=0.2, random_state=42)

        for img in val_imgs:
            src = os.path.join(src_folder, img)
            dst = os.path.join(val_dir, category, img)
            shutil.move(src, dst)

print("✅ Validation split created successfully!")
