from flask import Flask, render_template, request, jsonify, send_file
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import os
import io
import zipfile
import re

app = Flask(__name__)

# Load the pre-trained Stable Diffusion models and move to GPU
text_to_image_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
)
text_to_image_pipe = text_to_image_pipe.to("cuda")
text_to_image_pipe.enable_attention_slicing()

# Load the pre-trained Image-to-Image model
img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
)
img2img_pipe = img2img_pipe.to("cuda")
img2img_pipe.enable_attention_slicing()

# Load the pre-trained Image-to-Text (caption generation) model
image_to_text_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_to_text_model.to("cuda")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Art styles and configurations (from the old app.py)
ART_STYLES = [
    "Realistic Photography", "Watercolor", "Line Drawing", "Digital Art", "Graffiti", "Vincent van Gogh",
    "Comic Art", "Pixel Art", "Oil Painting", "Pop Art", "Psychedelic", "Modern Art", "Neon Art", "Sketch"
]

IMAGE_SIZES = {
    "1:1": (512, 512), "3:2": (768, 512), "4:3": (768, 576), "3:4": (576, 768), "16:9": (912, 512), "9:16": (512, 912)
}

RESOLUTIONS = {"low": 0.5, "medium": 0.8, "high": 1.3}


# Input validation
def is_valid_prompt(prompt):
    if re.match(r'^[0-9]+$', prompt) or re.match(r'^[a-zA-Z]+$', prompt):
        return False
    if len(prompt.split()) < 3:
        return False
    return True


# NSFW content detection (simple keyword-based approach)
def is_nsfw(prompt):
    nsfw_keywords = ['nude', 'naked', 'sex', 'porn', 'xxx', 'adult', 'explicit']
    return any(keyword in prompt.lower() for keyword in nsfw_keywords)


# Prompt optimization for text-to-image
def optimize_prompt(prompt, art_style):
    optimized_prompt = f"high quality, detailed, {art_style} style, {prompt}"
    optimized_prompt = optimized_prompt.replace("quality", "(quality:1.2)").replace("detailed", "(detailed:1.2)")
    negative_prompt = "blurry, low resolution, poorly drawn, bad anatomy, wrong proportions, extra limbs, disfigured, deformed, body out of frame, bad composition, watermark, signature, text"
    return optimized_prompt, negative_prompt


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        data = request.get_json()
        prompt = data["prompt"]
        num_images = int(data["num_images"])
        art_style = data["art_style"]
        image_size = data["image_size"]
        resolution = data["resolution"]

        if not is_valid_prompt(prompt):
            return jsonify({"error": "Please enter a valid prompt with at least 3 words."})

        if is_nsfw(prompt):
            return jsonify({"error": "Please enter a proper prompt without NSFW content."})

        optimized_prompt, negative_prompt = optimize_prompt(prompt, art_style)
        images = []

        base_width, base_height = IMAGE_SIZES[image_size]
        resolution_multiplier = RESOLUTIONS[resolution]
        width = int(base_width * resolution_multiplier)
        height = int(base_height * resolution_multiplier)

        for _ in range(num_images):
            image = text_to_image_pipe(
                prompt=optimized_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=50,
                guidance_scale=7.5,
                width=width,
                height=height
            ).images[0]
            images.append(image)

        if num_images == 1:
            image_path = os.path.join("static", "generated_image.png")
            images[0].save(image_path)
            return jsonify({"image_path": f"/static/generated_image.png"})
        else:
            image_paths = []
            for i, image in enumerate(images):
                image_path = os.path.join("static", f"generated_image_{i + 1}.png")
                image.save(image_path)
                image_paths.append(f"/static/generated_image_{i + 1}.png")
            return jsonify({"image_paths": image_paths})

    return render_template("index.html", art_styles=ART_STYLES)


@app.route("/image_to_image")
def image_to_image():
    return render_template("image_to_image.html", art_styles=ART_STYLES)


@app.route("/image_to_image", methods=["POST"])
def generate_image_from_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})

    file = request.files["image"]
    prompt = request.form.get("prompt", "")

    if not is_valid_prompt(prompt):
        return jsonify({"error": "Please enter a valid prompt with at least 3 words."})

    if is_nsfw(prompt):
        return jsonify({"error": "Please enter a proper prompt without NSFW content."})

    try:
        init_image = Image.open(file.stream).convert("RGB")
        init_image = init_image.resize((512, 512))

        optimized_prompt, negative_prompt = optimize_prompt(prompt, "")

        image = img2img_pipe(
            prompt=optimized_prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            strength=0.75,
            guidance_scale=7.5
        ).images[0]

        image_path = os.path.join("static", "generated_image.png")
        image.save(image_path)
        return jsonify({"image_path": f"/static/generated_image.png"})

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/image_to_text")
def image_to_text():
    return render_template("image_to_text.html")


@app.route("/image_to_text", methods=["POST"])
def generate_text_from_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})

    file = request.files["image"]

    try:
        image = Image.open(file.stream).convert("RGB")
        pixel_values = image_processor(image, return_tensors="pt").pixel_values.to("cuda")

        output_ids = image_to_text_model.generate(pixel_values, max_length=16, num_beams=4)
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return jsonify({"caption": caption})

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/download_images", methods=["POST"])
def download_images():
    data = request.get_json()
    image_paths = data["image_paths"]

    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        for i, path in enumerate(image_paths):
            image_path = os.path.join(app.root_path, path.lstrip('/'))
            zf.write(image_path, f"generated_image_{i + 1}.png")

    memory_file.seek(0)
    return send_file(memory_file, mimetype='application/zip', as_attachment=True, download_name='generated_images.zip')


if __name__ == "__main__":
    app.run(debug=True)
