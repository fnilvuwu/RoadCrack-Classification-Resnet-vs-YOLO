let selectedFiles = [];
// const resultListElement = document.getElementById('topRight');

// if (typeof yolo_results !== 'undefined' && yolo_results !== null && Object.keys(yolo_results.results).length > 1) {
//     resultListElement.classList.add('scrollable');
// } else {
//     resultListElement.classList.remove('scrollable');
// }

document.addEventListener('DOMContentLoaded', function () {
    const fileListElement = document.getElementById('fileList');
    const displayedImages = document.querySelectorAll('.displayedImage');
    const directoryInput = document.getElementById('directoryInput');



    directoryInput.addEventListener('change', function (event) {
        selectedFiles = Array.from(event.target.files);
        updateFileList();
    });

    document.getElementById('directoryForm').addEventListener('submit', function (event) {
        if (selectedFiles.length === 0) {
            event.preventDefault(); // Prevent default form submission
            alert('Please select a file before starting the classification process.');
            return;
        }

        // Iterate over selected files
        for (const file of selectedFiles) {
            // Check if the file is an image
            if (!file.type.startsWith('image/')) {
                event.preventDefault(); // Prevent default form submission
                alert('Only image files are allowed.');
                return;
            }
        }
    });

    function updateFileList() {
        fileListElement.innerHTML = '';

        if (selectedFiles.length === 0) {
            const noFileElement = document.createElement('li');
            noFileElement.textContent = 'No files selected.';
            noFileElement.classList.add('no-file');
            fileListElement.appendChild(noFileElement);
            return;
        }

        selectedFiles.forEach((file, index) => {
            const fileElement = document.createElement('li');
            fileElement.textContent = file.name;
            fileElement.addEventListener('click', () => {
                displayImage(file);
                highlightSelectedFile(fileElement);
            });
            fileListElement.appendChild(fileElement);
        });

        if (selectedFiles.length > 20) {
            fileListElement.classList.add('scrollable');
        } else {
            fileListElement.classList.remove('scrollable');
        }
    }

    function displayImage(file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            displayedImages[0].src = e.target.result;
            displayedImages[1].src = e.target.result;
        }
        reader.readAsDataURL(file);
    }

    function highlightSelectedFile(selectedElement) {
        Array.from(fileListElement.children).forEach(element => element.classList.remove('active'));
        selectedElement.classList.add('active');
    }

    document.getElementById('exportButton').addEventListener('click', function (event) {
        console.log("Export button clicked.");

        console.log("YOLO results:", yolo_results);
        console.log("TensorFlow results:", tf_results);

        // Check if either YOLO or TensorFlow results are available
        const hasResults = (yolo_results && Object.keys(yolo_results).length > 0) || (tf_results && Object.keys(tf_results).length > 0);

        // If there are no results, display an alert
        if (!hasResults) {
            alert("No classification results available. Please perform classification before exporting.");
            return;
        }

        // Make a POST request to the "/export_results" route
        fetch('/export_results', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                results_yolo: yolo_results,
                results_resnet: tf_results
            })
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                // Handle success response
                console.log("Export successful. Response:", data.message);
            })
            .catch(error => {
                // Handle error
                console.error('Error:', error);
            });
        alert("Result exported successfully.");
    });
});
