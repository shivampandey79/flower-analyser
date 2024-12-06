# Install necessary packages
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
import streamlit as st

# Load the 'tf_flowers' dataset and pre-trained model
@st.cache_resource
def load_model_and_dataset():
    # Use pre-trained MobileNetV2 model (Faster and efficient for image classification)
    base_model = tf.keras.applications.MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze base model layers for inference
    
    # Create a custom top layer
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')  # 5 classes in 'tf_flowers'
    ])
    
    # Compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Load dataset
    dataset, info = tfds.load('tf_flowers', with_info=True, as_supervised=True)
    
    # Shuffle dataset
    full_dataset = dataset['train'].shuffle(1000)
    
    # Image size and batch size
    IMG_SIZE = 128
    BATCH_SIZE = 32
    
    # Preprocessing function for image resizing and normalization
    def preprocess_image(image, label):
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)  # MobileNetV2 specific preprocessing
        return image, label
    
    # Preprocess datasets
    train_size = int(0.8 * info.splits['train'].num_examples)
    train_dataset = full_dataset.take(train_size).map(preprocess_image).batch(BATCH_SIZE)
    val_dataset = full_dataset.skip(train_size).map(preprocess_image).batch(BATCH_SIZE)
    
    # Train the model for quick testing
    model.fit(train_dataset, validation_data=val_dataset, epochs=3)  # Reduced epochs for faster testing
    
    return model, info

# Load model and dataset
st.title("🌸 Flower Recognition App 🌸")
model, info = load_model_and_dataset()

# Upload user image
st.header("Upload an Image")
uploaded_file = st.file_uploader("Choose a flower image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    IMG_SIZE = 128
    img = Image.open(uploaded_file).convert('RGB')  # Ensure the image is in RGB mode
    img = img.resize((IMG_SIZE, IMG_SIZE))  # Resize the image
    img_array = np.array(img)  # No need to normalize manually here
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make predictions
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    flower_species = info.features['label'].int2str(predicted_class_index)

    # Display prediction results
    st.success(f"🌷 **Predicted Flower Species: {flower_species}** 🌷")
    st.write(f"Confidence: {predictions[0][predicted_class_index]:.2%}")

    # Display probability chart
    st.subheader("Prediction Probabilities")
    prob_chart = {
        info.features['label'].int2str(i): prob
        for i, prob in enumerate(predictions[0])
    }
    st.bar_chart(prob_chart)
else:
    st.info("🖼️ Upload a flower image to get started!")
