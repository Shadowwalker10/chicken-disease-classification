document.addEventListener("DOMContentLoaded", function() {
    const predictBtn = document.getElementById("predict-btn");
    const imageInput = document.getElementById("image-input");
    const cameraInput = document.getElementById("camera-input");
    const cameraBtn = document.getElementById("camera-btn");
    const uploadedImage = document.getElementById("uploaded-image");
    const resultDiv = document.getElementById("result");

    function handleFileInput(file) {
        const reader = new FileReader();
        reader.onload = function() {
            let imageData = reader.result;

            // Strip the base64 metadata (e.g., "data:image/jpeg;base64,")
            imageData = imageData.replace(/^data:image\/[a-z]+;base64,/, "");

            const data = { image: imageData };

            // Display the uploaded image
            uploadedImage.src = `data:image/jpeg;base64,${imageData}`;
            uploadedImage.style.display = "block";  // Show the image

            fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(data => {
                const prediction = data.prediction;
                const confidence = data.confidence;
                resultDiv.innerHTML = `Prediction: ${prediction} (Confidence: ${confidence}%)`;
            })
            .catch(error => console.error(error));
        };
        reader.readAsDataURL(file);
    }

    // Handle the take picture button click by triggering the hidden file input
    cameraBtn.addEventListener("click", function() {
        imageInput.click(); // Opens the file input to take a picture or choose a file
    });

    // Handle image upload from the file input
    imageInput.addEventListener("change", function() {
        if (imageInput.files.length > 0) {
            handleFileInput(imageInput.files[0]);
        }
    });

    // Predict button click event
    predictBtn.addEventListener("click", function() {
        if (imageInput.files.length > 0) {
            handleFileInput(imageInput.files[0]);
        } else {
            alert("Please select an image or take a new picture");
        }
    });
});
