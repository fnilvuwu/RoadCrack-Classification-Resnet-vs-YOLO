from flask import Flask, render_template, request, jsonify
import platform
import psutil
import cpuinfo
import GPUtil
from classification import classify_image_yolo, classify_image_tf
import os
import pandas as pd
from werkzeug.utils import secure_filename

app = Flask(__name__)


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
    return jsonify(classification_results)


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

    # Return classification result as JSON
    return jsonify(classification_results)


# Route for classifying each uploaded image
@app.route("/classify_multi_yolo", methods=["POST"])
def classify_multi_yolo():
    if request.method == "POST":
        # Get the list of files from the webpage
        files = request.files.getlist("file")

        # Store classification results for each image
        classification_results = []

        # Initialize total accuracy
        total_accuracy = 0.0
        total_speed = 0

        # Iterate for each file in the files list, and classify them
        for file in files:
            image_path = os.path.join("temp/img/", file.filename)
            file.save(image_path)

            result = classify_image_yolo(image_path)
            classification_results.append(result)

            # Add accuracy to total
            total_accuracy += result[0]["accuracy"]
            total_speed += result[0]["inference_speed_ms"]

        # Calculate average accuracy
        if len(files) > 0:
            total_accuracy = total_accuracy / len(files)
        else:
            total_accuracy = total_accuracy

        total_accuracy = round(total_accuracy, 2)
        total_speed = round(total_speed, 2)
        # Add total accuracy to the classification results
        classification_results.append({"total_accuracy": total_accuracy})
        classification_results.append({"total_speed": total_speed})
        # Return classification results as JSON
        print(classification_results)
        return jsonify(classification_results)


@app.route("/classify_multi_tf", methods=["POST"])
def classify_multi_tf():
    if request.method == "POST":
        # Get the list of files from the webpage
        files = request.files.getlist("file")

        # Store classification results for each image
        classification_results = []

        # Initialize total accuracy
        total_accuracy = 0.0
        total_speed = 0

        # Iterate for each file in the files list, and classify them
        for file in files:
            image_path = os.path.join("temp/img/", file.filename)
            file.save(image_path)

            result = classify_image_tf(image_path)
            classification_results.append(result)

            # Add accuracy to total
            total_accuracy += result[0]["accuracy"]
            total_speed += result[0]["inference_time"]

        # Calculate average accuracy
        if len(files) > 0:
            total_accuracy = total_accuracy / len(files)
        else:
            total_accuracy = total_accuracy

        total_accuracy = round(total_accuracy, 2)
        total_speed = round(total_speed, 2)

        # Add average accuracy to the classification results
        classification_results.append({"total_accuracy": total_accuracy})
        classification_results.append({"total_speed": total_speed})
        # Return classification results as JSON
        print(classification_results)
        return jsonify(classification_results)


def export_to_excel(results_yolo, results_resnet):
    # Create DataFrames for YOLO and ResNet results
    df_yolo = pd.DataFrame(
        [result for sublist in results_yolo[:-2] for result in sublist]
    )
    df_resnet = pd.DataFrame(
        [result for sublist in results_resnet[:-2] for result in sublist]
    )

    # Extract total_accuracy and total_speed
    total_accuracy_yolo = results_yolo[-2]["total_accuracy"]
    total_speed_yolo = results_yolo[-1]["total_speed"]
    total_accuracy_resnet = results_resnet[-2]["total_accuracy"]
    total_speed_resnet = results_resnet[-1]["total_speed"]

    # Create DataFrames for total_accuracy and total_speed
    df_total_yolo = pd.DataFrame({"total_accuracy": [total_accuracy_yolo]})
    df_total_resnet = pd.DataFrame({"total_accuracy": [total_accuracy_resnet]})
    df_speed_yolo = pd.DataFrame({"total_speed": [total_speed_yolo]})
    df_speed_resnet = pd.DataFrame({"total_speed": [total_speed_resnet]})

    # Write DataFrames to Excel file on separate sheets
    with pd.ExcelWriter("classification_results.xlsx") as writer:
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


# Route to handle export request
@app.route("/export_results", methods=["POST"])
def export_results():
    results_yolo = request.json.get("results_yolo")
    results_resnet = request.json.get("results_resnet")
    print(results_yolo)
    print(results_resnet)

    # Call function to export results to Excel
    export_to_excel(results_yolo, results_resnet)

    # Return success response
    return jsonify({"message": "Classification results exported successfully"})


# Route for 'tunggal' page
@app.route("/tunggal")
def tunggal():
    return render_template("tunggal.html")


# Route for home page
@app.route("/")
def home():
    specs = get_system_info()
    return render_template("home.html", specs=specs)


# Route for 'multi' page
@app.route("/multi")
def multi():
    return render_template("multi.html")


# Route for 'test' page
@app.route("/test")
def test():

    return render_template("test.html")


if __name__ == "__main__":
    app.run(debug=True)
