import cv2
import os
from ultralytics import YOLO
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import time


def classify_image_yolo(image_path):
    # Initialize the model
    model = YOLO("best.pt", task="classify")
    results = model.predict(source=image_path)

    classification_results = {}

    for result in results:
        # Read the image
        image = cv2.imread(image_path)

        # Convert speed values to milliseconds (ms)
        preprocess_speed_ms = round(result.speed["preprocess"], 1)
        inference_speed_ms = round(result.speed["inference"], 1)
        postprocess_speed_ms = round(result.speed["postprocess"], 1)

        # Get the predicted class label
        predicted_class_index = result.probs.top1
        predicted_class_label = result.names[predicted_class_index]

        # Get the confidence
        confidence = result.probs.top1conf.item() * 100
        confidence = round(confidence, 2)

        filename = os.path.basename(image_path)
        filename_lowercase = filename.lower()

        true_class_label = None
        if "good" in filename_lowercase:
            true_class_label = "good"
        elif "pothole" in filename_lowercase:
            true_class_label = "pothole"
        elif "crack" in filename_lowercase:
            true_class_label = "crack"

        # Check if predicted class matches true class and calculate accuracy
        if predicted_class_label == true_class_label:
            accuracy = 1
        else:
            accuracy = 0

        # Overlay the predicted class label, confidence, and accuracy on the image
        text = f"Predicted Class: {predicted_class_label} (Confidence: {confidence}%, Accuracy: {accuracy}%)"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        # Calculate the position to center the text
        text_x = (image.shape[1] - text_size[0]) // 2
        text_y = image.shape[0] - 20  # Position it 20 pixels from the bottom
        # Put the text on the image
        cv2.putText(
            image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3
        )

        # Save the image with the predicted class label
        image_name = f"predicted_{predicted_class_label}-" + os.path.basename(
            image_path
        )
        output_image_path = os.path.join("static", "result", "yolo", image_name)
        cv2.imwrite(output_image_path, image)

        # Store the classification results
        classification_results = {
            "image_path": output_image_path,
            "predicted_class": predicted_class_label,
            "actual_class": true_class_label,
            "confidence": confidence,
            "accuracy": accuracy,
            "preprocess_speed_ms": preprocess_speed_ms,
            "inference_speed_ms": inference_speed_ms,
            "postprocess_speed_ms": postprocess_speed_ms,
        }

    return classification_results


def classify_image_tf(image_path):
    # Load the model
    model_path = "savedModel"  # Replace with the actual model path
    model = hub.KerasLayer(model_path)

    class_names = ["pothole", "good", "crack"]
    img = Image.open(image_path).resize((224, 224))
    img = img.convert("RGB")  # Ensure the image is converted to RGB
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Start measuring time before inference
    start_time = time.time()
    predictions = model(img)

    # End measuring time after inference
    end_time = time.time()

    # Calculate inference time
    inference_time = round((end_time - start_time) * 1000, 1)  # Convert to milliseconds

    # Initialize an empty dictionary to store classification results
    classification_results = {}

    for prediction in predictions:
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_names[predicted_class_index]
        confidence = float(
            prediction[predicted_class_index] * 100
        )  # Convert to Python float

        confidence = round(confidence, 2)

        filename = os.path.basename(image_path)
        filename_lowercase = filename.lower()

        true_class_label = None
        if "good" in filename_lowercase:
            true_class_label = "good"
        elif "pothole" in filename_lowercase:
            true_class_label = "pothole"
        elif "crack" in filename_lowercase:
            true_class_label = "crack"

        # Check if predicted class matches true class and calculate accuracy
        if predicted_class == true_class_label:
            accuracy = 1
        else:
            accuracy = 0

        # Overlay the predicted class label, confidence, and accuracy on the image
        image = cv2.imread(image_path)
        text = f"Predicted Class: {predicted_class} (Confidence: {confidence}%, Accuracy: {accuracy}%)"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        # Calculate the position to center the text
        text_x = (image.shape[1] - text_size[0]) // 2
        text_y = image.shape[0] - 20  # Position it 20 pixels from the bottom
        # Put the text on the image
        cv2.putText(
            image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3
        )

        # Save the image with the predicted class label
        image_name = f"predicted_{predicted_class}-" + os.path.basename(image_path)
        output_image_path = os.path.join("static", "result", "resnet", image_name)
        cv2.imwrite(output_image_path, image)
        # Append the classification result to the dictionary
        classification_results = {
            "image_path": output_image_path,
            "predicted_class": predicted_class,
            "actual_class": true_class_label,
            "confidence": confidence,
            "accuracy": accuracy,
            "inference_time": inference_time,
        }

    return classification_results
