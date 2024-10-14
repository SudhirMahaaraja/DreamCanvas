document.addEventListener("DOMContentLoaded", () => {
    const textToImageBtn = document.getElementById("text-to-image-btn");
    const generateBtn = document.getElementById("generate-btn");
    const saveBtn = document.getElementById("save-btn");
    const generatedImagesContainer = document.getElementById("generated-images");
    const imageContainer = document.getElementById("image-container");
    const loadingAnimation = document.getElementById("loading-animation");
    const artStyleButtons = document.querySelectorAll(".art-style-btn");
    const imageSizeButtons = document.querySelectorAll(".image-size-btn");
    const imageResolutionButtons = document.querySelectorAll(".image-resolution-btn");

    let selectedArtStyle = '';
    let selectedImageSize = '1:1';
    let selectedResolution = 'medium';

    // Handle art style selection
    artStyleButtons.forEach(button => {
        button.addEventListener("click", () => {
            artStyleButtons.forEach(btn => btn.classList.remove("selected"));
            button.classList.add("selected");
            selectedArtStyle = button.dataset.style;
        });
    });

    // Handle image size selection
    imageSizeButtons.forEach(button => {
        button.addEventListener("click", () => {
            imageSizeButtons.forEach(btn => btn.classList.remove("selected"));
            button.classList.add("selected");
            selectedImageSize = button.dataset.size;
        });
    });

    // Handle image resolution selection
    imageResolutionButtons.forEach(button => {
        button.addEventListener("click", () => {
            imageResolutionButtons.forEach(btn => btn.classList.remove("selected"));
            button.classList.add("selected");
            selectedResolution = button.dataset.resolution;
        });
    });

    // Trigger the generation with both 'Enter' key and mouse click
    document.getElementById("prompt").addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            generateBtn.click();
        }
    });

    // Function to speak text with a female voice
    function speakWithFemaleVoice(text) {
        const utterance = new SpeechSynthesisUtterance(text);
        const voices = speechSynthesis.getVoices();
        const femaleVoice = voices.find(voice => voice.name.includes('Female') && voice.lang.startsWith('en'));

        if (femaleVoice) {
            utterance.voice = femaleVoice;
        }

        utterance.pitch = 1.2;
        utterance.rate = 0.9;
        utterance.volume = 0.8;
        speechSynthesis.speak(utterance);
    }

    // Handle image generation
    generateBtn.addEventListener("click", () => {
        const prompt = document.getElementById("prompt").value;
        const numImages = document.querySelector('input[name="num_images"]:checked').value;

        if (prompt.trim() === "") {
            alert("Please enter a prompt!");
            return;
        }

        // Read aloud the prompt before generating the image
        speakWithFemaleVoice(`Generating ${numImages} image${numImages > 1 ? 's' : ''} of ${prompt} in ${selectedArtStyle} style, with size ${selectedImageSize} and ${selectedResolution} resolution.`);

        // Show loading animation
        loadingAnimation.style.display = "block";
        generatedImagesContainer.innerHTML = '';
        imageContainer.style.display = "block";
        saveBtn.style.display = "none";

        // Make a request to the Flask server for image generation
        fetch("/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                prompt,
                num_images: numImages,
                art_style: selectedArtStyle,
                image_size: selectedImageSize,
                resolution: selectedResolution
            })
        })
        .then(response => response.json())
        .then(data => {
            loadingAnimation.style.display = "none";
            if (data.error) {
                alert(data.error);
            } else if (data.image_paths) {
                // Multiple images
                data.image_paths.forEach(path => {
                    const img = document.createElement('img');
                    img.src = path;
                    generatedImagesContainer.appendChild(img);
                });
                saveBtn.textContent = 'Download Images';
                saveBtn.style.display = "inline-block";
                saveBtn.onclick = () => {
                    fetch('/download_images', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image_paths: data.image_paths })
                    })
                    .then(response => response.blob())
                    .then(blob => {
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.style.display = 'none';
                        a.href = url;
                        a.download = 'generated_images.zip';
                        document.body.appendChild(a);
                        a.click();
                        window.URL.revokeObjectURL(url);
                    });
                };
            } else if (data.image_path) {
                // Single image
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