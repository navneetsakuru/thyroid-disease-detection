import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import numpy as np
import os

# ===============================
# Load Model (your old setup)
# ===============================
def create_model(input_shape=(224, 224, 3)):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', name='conv2d_1'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', name='conv2d_2'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', name='conv2d_3'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def load_model(model_path='thyroid_model.h5'):
    """Load the trained model or fall back to fresh model."""
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Loaded model from {model_path}")
    except Exception:
        model = create_model()
        print("⚠️ No pre-trained model found. Using untrained model.")
    return model


# ===============================
# Prediction Helpers
# ===============================
def predict_image(model, image_array):
    prediction = model.predict(image_array, verbose=0)[0][0]
    label = 'Malignant' if prediction < 0.5 else 'Benign'
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return label, confidence


def preprocess_image(image, target_size=(224, 224)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def add_bounding_box(image, label, confidence):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    margin = 0.2
    x0, y0 = width * margin, height * margin
    x1, y1 = width * (1 - margin), height * (1 - margin)

    box_color = "red" if label == "Malignant" else "pink"
    draw.rectangle([x0, y0, x1, y1], outline=box_color, width=5)

    try:
        font = ImageFont.truetype("arial.ttf", size=int(height * 0.05))
    except:
        font = ImageFont.load_default()

    text = f"{label} ({confidence:.1%})"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    text_x, text_y = x1 - text_w - 10, y0 + 10

    draw.rectangle([text_x-5, text_y-5, text_x+text_w+5, text_y+text_h+5], fill="black")
    draw.text((text_x, text_y), text, fill="white", font=font)

    return image


# ===============================
# Streamlit App (your original UI)
# ===============================
def main():
    st.title("Thyroid Nodule Detection Dashboard")
    st.write("Upload an ultrasound image to classify thyroid nodules as malignant or benign.")

    uploaded_file = st.file_uploader("Choose an ultrasound image...", type=["jpg", "png"])
    if uploaded_file is not None:
        os.makedirs('data/uploads', exist_ok=True)
        image_path = f"data/uploads/{uploaded_file.name}"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        original_image = Image.open(image_path)
        image_array = preprocess_image(original_image)

        model = load_model()
        label, confidence = predict_image(model, image_array)
        annotated_image = add_bounding_box(original_image.copy(), label, confidence)

        col1, col2 = st.columns([3, 1])
        with col1:
            st.image(annotated_image, caption="Annotated Ultrasound Image", use_container_width=True)
        with col2:
            st.subheader("Prediction Results")
            st.metric("Diagnosis", label)
            st.metric("Confidence", f"{confidence:.1%}")
            st.markdown("---")
            st.subheader("Interpretation")
            if label == "Malignant":
                st.error("This nodule shows characteristics associated with malignancy. Please consult a specialist.")
            else:
                st.success("This nodule appears benign. Regular follow-up may still be recommended.")


if __name__ == "__main__":
    main()
