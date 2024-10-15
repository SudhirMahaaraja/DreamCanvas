document.addEventListener("DOMContentLoaded", () => {
    const imageUpload = document.getElementById("image-upload");
    const generateBtn = document.getElementById("generate-btn");
    const resultContainer = document.getElementById("result-container");
    const loadingAnimation = document.getElementById("loading-animation");
    const generatedCaption = document.getElementById("generated-caption");

    generateBtn.addEventListener("click", () => {
        const file = imageUpload.files[0];

        if (!file) {
            alert("Please upload an image!");
            return;
        }

        const formData = new FormData();
        formData.append("image", file);

        loadingAnimation.style.display = "block";
        resultContainer.style.display = "block";
        generatedCaption.textContent = '';

        fetch("/image_to_text", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            loadingAnimation.style.display = "none";
            if (data.error) {
                alert(data.error);
            } else if (data.caption) {
                generatedCaption.textContent = data.caption;
            }
        })
        .catch(error => {
            loadingAnimation.style.display = "none";
            console.error("Error generating caption:", error);
            alert("An error occurred while generating the caption. Please try again.");
        });
    });
});