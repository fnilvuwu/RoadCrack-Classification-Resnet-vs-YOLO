import os
from PIL import Image
import shutil

# Function to check if a file is a valid JPEG image
def is_valid_jpeg(file_path):
    try:
        # Attempt to open the file as a JPEG image
        with Image.open(file_path) as img:
            img.verify()
            return True
    except IsADirectoryError:
        # If the file is a directory, skip it
        print(f"Skipping directory: {file_path}")
        return False
    except UnidentifiedImageError:
        # If the file is not recognized as an image, print a warning and return False
        print(f"File is not recognized as an image: {file_path}")
        return False
    except Exception as e:
        # If an error occurs, the file is not a valid JPEG image
        print(f"Invalid JPEG image: {file_path} ({e})")
        return False

# Function to preprocess images in a folder
def preprocess_images(folder_path, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        try:
            # Check if the file is a valid JPEG image
            if is_valid_jpeg(os.path.join(folder_path, filename)):
                # If the file is a valid JPEG image, preprocess the image (if needed)
                # For now, just copy the image to the output folder
                output_path = os.path.join(output_folder, filename)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                shutil.copyfile(os.path.join(folder_path, filename), output_path)

                # Print a message indicating successful preprocessing
                print(f"Preprocessed {filename}")
        except Exception as e:
            # If an error occurs during preprocessing, skip the image
            print(f"Skipping {filename}: {e}")

# Define the paths to the folders containing the images
folders = [
    ("/home/fnilvu/Downloads/Proyek Skripsi/RoadCrack Flask/data/train/crack", "preprocessed_images/train/crack"),
    ("/home/fnilvu/Downloads/Proyek Skripsi/RoadCrack Flask/data/train/good", "preprocessed_images/train/good"),
    ("/home/fnilvu/Downloads/Proyek Skripsi/RoadCrack Flask/data/train/pothole", "preprocessed_images/train/pothole"),
    ("/home/fnilvu/Downloads/Proyek Skripsi/RoadCrack Flask/data/val/crack", "preprocessed_images/val/crack"),
    ("/home/fnilvu/Downloads/Proyek Skripsi/RoadCrack Flask/data/val/good", "preprocessed_images/val/good"),
    ("/home/fnilvu/Downloads/Proyek Skripsi/RoadCrack Flask/data/val/pothole", "preprocessed_images/val/pothole")
]

# Preprocess images in each folder
for folder_path, output_folder in folders:
    preprocess_images(folder_path, output_folder)
