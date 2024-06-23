from flask import Flask, render_template, request, jsonify, send_from_directory
import platform
import psutil
import cpuinfo
import GPUtil
from classification import classify_image_yolo, classify_image_tf
import os
import pandas as pd
from werkzeug.utils import secure_filename
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from openpyxl import Workbook
from openpyxl.drawing.image import Image

app = Flask(__name__)
matplotlib.use(
    "Agg"
)  # Set the backend to 'Agg' fix UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.


# Function to get system information
def get_system_info():
    system_info = {"Platform": platform.system()}

    cpu_info = cpuinfo.get_cpu_info()
    system_info["CPU"] = cpu_info["brand_raw"]

    system_info["Cores"] = psutil.cpu_count(logical=False)

    ram = psutil.virtual_memory().total / (1024.0**2)  # Total RAM in MB
    system_info["RAM"] = str(round(ram)) + " MB"

    gpus = GPUtil.getGPUs()
    if gpus:
        gpu_list = []
        for gpu_info in gpus:
            gpu = {"Name": gpu_info.name, "GPU RAM": gpu_info.memoryTotal}
            gpu_list.append(gpu)
        system_info["GPUs"] = gpu_list
    else:
        system_info["GPUs"] = "N/A"

    return system_info


# Route for home page
@app.route("/")
def home():
    specs = get_system_info()
    return render_template("home.html", specs=specs)


@app.route("/classify_yolo", methods=["POST"])
def classify():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"})

    # Save the uploaded file
    filename = secure_filename(file.filename)
    file.save(filename)

    # Perform classification
    classification_results = classify_image_yolo(filename)

    # Assuming classification_results is a dictionary
    # Adjust the response to match the new format
    response = {
        "results": {
            "yolo_1": classification_results  # Assuming only one result for YOLO
        },
        "total_accuracy": classification_results[
            "accuracy"
        ],  # Assuming accuracy is directly available in the result
        "total_speed": classification_results[
            "inference_speed_ms"
        ],  # Assuming inference speed is directly available
    }
    return jsonify(response)


@app.route("/classify_tf", methods=["POST"])
def classify_tf():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"})

    # Save the uploaded file
    filename = secure_filename(file.filename)
    file.save(filename)

    # Perform classification using TensorFlow
    classification_results = classify_image_tf(filename)

    # Assuming classification_results is a dictionary
    # Adjust the response to match the new format
    response = {
        "results": {
            "tf_1": classification_results  # Assuming only one result for TensorFlow
        },
        "total_accuracy": classification_results[
            "accuracy"
        ],  # Assuming accuracy is directly available in the result
        "total_speed": classification_results[
            "inference_time"
        ],  # Assuming inference speed is directly available
    }
    return jsonify(response)


def export_to_excel(
    results_yolo, results_resnet, yolo_conf_matrix_path, tf_conf_matrix_path
):
    try:
        # Ensure that results_yolo["results"] and results_resnet["results"] contain lists of dictionaries
        yolo_results_list = list(results_yolo["results"].values())
        resnet_results_list = list(results_resnet["results"].values())

        # Create DataFrames for YOLO and ResNet results
        df_yolo = pd.DataFrame(yolo_results_list)
        df_resnet = pd.DataFrame(resnet_results_list)

        # Extract total_accuracy and total_speed
        total_accuracy_yolo = results_yolo["total_accuracy"]
        total_speed_yolo = results_yolo["total_speed"]
        total_accuracy_resnet = results_resnet["total_accuracy"]
        total_speed_resnet = results_resnet["total_speed"]

        # Create DataFrames for total_accuracy and total_speed
        df_total_yolo = pd.DataFrame({"total_accuracy": [total_accuracy_yolo]})
        df_total_resnet = pd.DataFrame({"total_accuracy": [total_accuracy_resnet]})
        df_speed_yolo = pd.DataFrame({"total_speed": [total_speed_yolo]})
        df_speed_resnet = pd.DataFrame({"total_speed": [total_speed_resnet]})

        # Write DataFrames to Excel file on separate sheets
        with pd.ExcelWriter("classification_results.xlsx", engine="openpyxl") as writer:
            df_yolo.to_excel(writer, sheet_name="YOLO Results", index=False)
            df_resnet.to_excel(writer, sheet_name="ResNet Results", index=False)
            df_total_yolo.to_excel(
                writer,
                sheet_name="YOLO Results",
                startcol=df_yolo.shape[1] + 2,
                index=False,
            )
            df_speed_yolo.to_excel(
                writer,
                sheet_name="YOLO Results",
                startcol=df_yolo.shape[1] + 3,
                index=False,
            )
            df_total_resnet.to_excel(
                writer,
                sheet_name="ResNet Results",
                startcol=df_resnet.shape[1] + 2,
                index=False,
            )
            df_speed_resnet.to_excel(
                writer,
                sheet_name="ResNet Results",
                startcol=df_resnet.shape[1] + 3,
                index=False,
            )

            # Add YOLOv8 confusion matrix to a new sheet
            wb = writer.book
            img = Image(yolo_conf_matrix_path)
            wb.create_sheet(title="YOLO Confusion Matrix")
            ws = wb["YOLO Confusion Matrix"]
            ws.add_image(img)

            # Add ResNet-50 confusion matrix to a new sheet
            img = Image(tf_conf_matrix_path)
            wb.create_sheet(title="ResNet Confusion Matrix")
            ws = wb["ResNet Confusion Matrix"]
            ws.add_image(img)

    except Exception as e:
        print(f"An error occurred: {e}")
        raise



# Route to handle export request
@app.route("/export_results", methods=["POST"])
def export_results():
    results_yolo = request.json.get("results_yolo")
    results_resnet = request.json.get("results_resnet")
    yolo_conf_matrix_path = "static/result/confusion_matrix/yolo_confusion_matrix.png"
    tf_conf_matrix_path = "static/result/confusion_matrix/tf_confusion_matrix.png"

    # Call function to export results to Excel
    export_to_excel(
        results_yolo, results_resnet, yolo_conf_matrix_path, tf_conf_matrix_path
    )

    # Return success response
    return jsonify({"message": "Classification results exported successfully"})


# Route for 'tunggal' page
@app.route("/tunggal")
def tunggal():
    return render_template("tunggal.html")


def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_confusion_matrix(cm, labels, save_path, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        fmt="d",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()

    # Ensure the directory exists
    ensure_directory(os.path.dirname(save_path))

    # Save the plot as an image
    fig.savefig(save_path)
    plt.close(fig)


def compute_confusion_matrix_yolo(predictions_dict, labels):
    pred_labels = [
        result["predicted_class"] for result in predictions_dict["results"].values()
    ]
    actual_labels = [
        result["actual_class"] for result in predictions_dict["results"].values()
    ]

    print(f"actual label: {actual_labels}")
    print(f"pred label: {pred_labels}")

    cm = confusion_matrix(actual_labels, pred_labels, labels=labels)
    return cm


def compute_confusion_matrix_tf(predictions_dict, labels):
    pred_labels = [
        result["predicted_class"] for result in predictions_dict["results"].values()
    ]
    actual_labels = [
        result["actual_class"] for result in predictions_dict["results"].values()
    ]

    print(f"actual label: {actual_labels}")
    print(f"pred label: {pred_labels}")

    cm = confusion_matrix(actual_labels, pred_labels, labels=labels)
    return cm


@app.route("/multi", methods=["GET", "POST"])
def multi():
    file_locations = {}

    if request.method == "POST":
        # Check if files are present in the request
        if "file" not in request.files:
            error_message = f"No file is detected. Please select a file."
            return jsonify({"error": error_message})

        # Get the list of files from the form
        files = request.files.getlist("file")

        # Store the uploaded files
        # After processing the files, you can pass them to your classification functions
        yolo_results = {"results": {}}
        tf_results = {"results": {}}
        file_locations = {}

        counter = 1

        # Check if filename contains keywords
        for file in files:
            if not any(
                keyword in file.filename.lower()
                for keyword in ["pothole", "good", "crack"]
            ):
                error_message = f"Filename '{file.filename}' does not contain 'pothole', 'good', or 'crack'. Prediction won't be running."
                return jsonify({"error": error_message})

        # Loop over the files for YOLO
        for file in files:
            image_path = os.path.join("temp/img/", file.filename)
            file_locations[counter] = image_path
            file.save(image_path)

            # Check if filename contains keywords
            if not any(
                keyword in file.filename.lower()
                for keyword in ["pothole", "good", "crack"]
            ):
                error_message = f"Filename '{file.filename}' does not contain 'pothole', 'good', or 'crack'. Prediction won't be running."
                return jsonify({"error": error_message})

            yolo_result = classify_image_yolo(image_path)
            yolo_results["results"][f"yolo_{counter}"] = yolo_result

            tf_result = classify_image_tf(image_path)
            tf_results["results"][f"tf_{counter}"] = tf_result

            counter += 1

        print(yolo_results)
        print(tf_results)
        labels = ["good", "pothole", "crack"]
        yolo_conf_matrix = compute_confusion_matrix_yolo(yolo_results, labels=labels)
        tf_conf_matrix = compute_confusion_matrix_tf(tf_results, labels=labels)

        yolo_cm_path = os.path.join(
            "static/result/confusion_matrix", "yolo_confusion_matrix.png"
        )
        tf_cm_path = os.path.join(
            "static/result/confusion_matrix", "tf_confusion_matrix.png"
        )
        plot_confusion_matrix(
            yolo_conf_matrix,
            labels=labels,
            save_path=yolo_cm_path,
            title="YOLOv8 Confusion Matrix",
        )
        plot_confusion_matrix(
            tf_conf_matrix,
            labels=labels,
            save_path=tf_cm_path,
            title="ResNet-50 Confusion Matrix",
        )

        # Calculate average accuracy and total speed for YOLO
        total_accuracy_yolo = sum(
            result["accuracy"] for result in yolo_results["results"].values()
        ) / len(yolo_results["results"])
        total_speed_yolo = sum(
            result["inference_speed_ms"] for result in yolo_results["results"].values()
        )

        # Calculate average accuracy and total speed for TensorFlow
        total_accuracy_tf = sum(
            result["accuracy"] for result in tf_results["results"].values()
        ) / len(tf_results["results"])
        total_speed_tf = sum(
            result["inference_time"] for result in tf_results["results"].values()
        )

        # Update total_accuracy and total_speed in yolo_results and tf_results
        yolo_results["total_accuracy"] = round(total_accuracy_yolo, 2)
        yolo_results["total_speed"] = round(total_speed_yolo, 2)
        tf_results["total_accuracy"] = round(total_accuracy_tf, 2)
        tf_results["total_speed"] = round(total_speed_tf, 2)
        print(file_locations)
        # Render the template with the results
        return render_template(
            "multi.html",
            yolo_results=yolo_results,
            tf_results=tf_results,
            file_locations=file_locations,
            yolo_conf_matrix=yolo_conf_matrix,
            tf_conf_matrix=tf_conf_matrix,
        )
    else:
        return render_template("multi.html", file_locations=file_locations)


@app.route("/multi/resnet/<path:image_filename>")
def get_image_resnet(image_filename):
    # Assuming the images are located in the static/img directory
    return send_from_directory("static/result/resnet", image_filename)


@app.route("/multi/yolo/<path:image_filename>")
def get_image_yolo(image_filename):
    # Assuming the images are located in the static/img directory
    return send_from_directory("static/result/yolo", image_filename)


if __name__ == "__main__":
    app.run(debug=True)
