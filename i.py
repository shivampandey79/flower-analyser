import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
from tkinter import Tk, filedialog

# Load the 'tf_flowers' dataset
dataset, info = tfds.load('tf_flowers', with_info=True, as_supervised=True)

# Get the dataset and shuffle
full_dataset = dataset['train'].shuffle(1000)

# Split the dataset (80% train, 20% validation)
train_size = int(0.8 * info.splits['train'].num_examples)
train_dataset = full_dataset.take(train_size)
val_dataset = full_dataset.skip(train_size)

# Image size and batch size
IMG_SIZE = 128
BATCH_SIZE = 32

# Preprocessing function to resize and normalize
def preprocess_image(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    return image, label

# Preprocess datasets
train_dataset = train_dataset.map(preprocess_image).batch(BATCH_SIZE)
val_dataset = val_dataset.map(preprocess_image).batch(BATCH_SIZE)

# Build the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(info.features['label'].num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=5)

# Function to predict flower species from user-selected images
def predict_flower_species():
    # Use tkinter to select an image file
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg *.png")])
    
    if file_path:
        # Load and preprocess the image
        img = image.load_img(file_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize and expand dims
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        flower_species = info.features['label'].int2str(predicted_class_index)
        
        # Display the image and prediction
        plt.imshow(img)
        plt.title(f"Predicted Flower Species: {flower_species}")
        plt.axis('off')
        plt.show()
        
        print(f"The model predicts this is a '{flower_species}'.")
    else:
        print("No file was selected.")

# Run the prediction function
predict_flower_species()
