import os
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub

# Load the model
model_path = 'savedModel'  # Replace with the actual model path
model = hub.KerasLayer(model_path)

class_names = ['good', 'pothole', 'crack']

# Specify the target directory for images
target_dir = 'temp/img'

def predict_image(image_path):
    img = Image.open(image_path).resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    predictions = model(img)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]
    accuracy = predictions[0][predicted_class_index] * 100

    return predicted_class, accuracy

if __name__ == '__main__':
    # Get a list of all image files in the target directory
    image_files = [f for f in os.listdir(target_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(target_dir, image_file)
        predicted_class, confidence = predict_image(image_path)
        print(f"Image: {image_file}")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence}%")
        print()