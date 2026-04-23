import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("thyroid_model.h5")

# Input folder containing all images
input_folder = "all_images"

# Output folders
benign_folder = "benign_samples"
malignant_folder = "malignant_samples"

os.makedirs(benign_folder, exist_ok=True)
os.makedirs(malignant_folder, exist_ok=True)

# Counters for renaming
benign_count = 1
malignant_count = 1

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        filepath = os.path.join(input_folder, filename)

        # Load & preprocess image
        img = Image.open(filepath).convert("RGB")
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict
        prediction = model.predict(img_array)[0][0]

        if prediction > 0.5:  # malignant
            new_name = f"malignant{malignant_count}.jpg"
            malignant_count += 1
            shutil.copy(filepath, os.path.join(malignant_folder, new_name))
            print(f"{filename} → {new_name} (Malignant)")
        else:  # benign
            new_name = f"benign{benign_count}.jpg"
            benign_count += 1
            shutil.copy(filepath, os.path.join(benign_folder, new_name))
            print(f"{filename} → {new_name} (Benign)")
