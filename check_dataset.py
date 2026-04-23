import os

# your dataset folder is called "dataset thyroid"
base_dir = "./dataset thyroid"

for split in ["train", "test", "validation"]:
    split_dir = os.path.join(base_dir, split)
    if os.path.exists(split_dir):
        benign_count = len(os.listdir(os.path.join(split_dir, "benign")))
        malignant_count = len(os.listdir(os.path.join(split_dir, "malignant")))
        print(f"{split.capitalize()} set distribution:")
        print(f"  Benign: {benign_count}")
        print(f"  Malignant: {malignant_count}")
        print("-" * 30)
    else:
        print(f"❌ {split} folder not found")
