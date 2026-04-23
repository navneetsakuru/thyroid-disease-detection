import os, random, shutil

test_dir = "./dataset thyroid/test"
benign_dir = os.path.join(test_dir, "benign")
malignant_dir = os.path.join(test_dir, "malignant")

benign_count = len(os.listdir(benign_dir))
malignant_count = len(os.listdir(malignant_dir))

print("Before balancing:")
print("Benign:", benign_count)
print("Malignant:", malignant_count)

# Keep the smaller number (benign)
target_count = benign_count

# Randomly keep only target_count malignant images
malignant_images = os.listdir(malignant_dir)
random.shuffle(malignant_images)

to_remove = malignant_images[target_count:]

# Move excess malignant images to backup folder
backup_dir = "./dataset/test_backup_malignant"
os.makedirs(backup_dir, exist_ok=True)

for img in to_remove:
    shutil.move(os.path.join(malignant_dir, img), os.path.join(backup_dir, img))

print("\nAfter balancing:")
print("Benign:", len(os.listdir(benign_dir)))
print("Malignant:", len(os.listdir(malignant_dir)))
print(f"Moved {len(to_remove)} malignant images to {backup_dir}")
