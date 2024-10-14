document.addEventListener("DOMContentLoaded", () => {
    const imageUpload = document.getElementById("image-upload");
    const generateBtn = document.getElementById("generate-btn");
    const saveBtn = document.getElementById("save-btn");
    const generatedImagesContainer = document.getElementById("generated-images");
    const imageContainer = document.getElementById("image-container");
    const loadingAnimation = document.getElementById("loading-animation");

    generateBtn.addEventListener("click", () => {
        const prompt = document.getElementById("prompt").value;
        const file = imageUpload.files[0];

        if (!file) {
            alert("Please upload an image!");
            return;
        }

        if (prompt.trim() === "") {
            alert("Please enter a prompt!");
            return;
        }

        const formData = new FormData();
        formData.append("image", file);
        formData.append("prompt", prompt);

        loadingAnimation.style.display = "block";
        generatedImagesContainer.innerHTML = '';
        imageContainer.style.display = "block";
        saveBtn.style.display = "none";

        fetch("/image_to_image", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            loadingAnimation.style.display = "none";
            if (data.error) {
                alert(data.error);
            } else if (data.image_path) {
                const img = document.createElement('img');
                img.src = data.image_path;
                generatedImagesContainer.appendChild(img);
                saveBtn.textContent = 'Save Image';
                saveBtn.style.display = "inline-block";
                saveBtn.onclick = () => {
                    const a = document.createElement('a');
                    a.href = data.image_path;
                    a.download = 'generated_image.png';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                };
            }
        })
        .catch(error => {
            loadingAnimation.style.display = "none";
            console.error("Error generating image:", error);
            alert("An error occurred while generating the image. Please try again.");
        });
    });
});