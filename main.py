import streamlit as st
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import io
import re

# Set the page config as the first Streamlit command
st.set_page_config(page_title="DreamCanvas", page_icon="🎨", layout="wide")

# Load models
@st.cache_resource
def load_models():
    text_to_image_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
    ).to("cuda")
    text_to_image_pipe.enable_attention_slicing()

    img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
    ).to("cuda")
    img2img_pipe.enable_attention_slicing()

    image_to_text_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to("cuda")
    image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    return text_to_image_pipe, img2img_pipe, image_to_text_model, image_processor, tokenizer

text_to_image_pipe, img2img_pipe, image_to_text_model, image_processor, tokenizer = load_models()

# Constants
ART_STYLES = [
    "Realistic Photography", "Watercolor", "Line Drawing", "Digital Art", "Graffiti", "Vincent van Gogh",
    "Comic Art", "Pixel Art", "Oil Painting", "Pop Art", "Psychedelic", "Modern Art", "Neon Art", "Sketch"
]

IMAGE_SIZES = {
    "1:1": (512, 512), "3:2": (768, 512), "4:3": (768, 576), "3:4": (576, 768), "16:9": (912, 512), "9:16": (512, 912)
}

RESOLUTIONS = {"low": 0.5, "medium": 0.8, "high": 1.3}

# Helper functions
def is_valid_prompt(prompt):
    if re.match(r'^[0-9]+$', prompt) or re.match(r'^[a-zA-Z]+$', prompt):
        return False
    if len(prompt.split()) < 3:
        return False
    return True

def is_nsfw(prompt):
    nsfw_keywords = ['nude', 'naked', 'sex', 'porn', 'xxx', 'adult', 'explicit']
    return any(keyword in prompt.lower() for keyword in nsfw_keywords)

def optimize_prompt(prompt, art_style):
    optimized_prompt = f"high quality, detailed, {art_style} style, {prompt}"
    optimized_prompt = optimized_prompt.replace("quality", "(quality:1.2)").replace("detailed", "(detailed:1.2)")
    negative_prompt = "blurry, low resolution, poorly drawn, bad anatomy, wrong proportions, extra limbs, disfigured, deformed, body out of frame, bad composition, watermark, signature, text"
    return optimized_prompt, negative_prompt

# Streamlit app
def main():
    st.title("DreamCanvas")

    # Sidebar navigation
    page = st.sidebar.selectbox("Choose a page", ["Text to Image", "Image to Image", "Image to Text"])

    if page == "Text to Image":
        text_to_image()
    elif page == "Image to Image":
        image_to_image()
    else:
        image_to_text()

def text_to_image():
    st.header("Text to Image")
    st.write("Welcome to our Text-to-Image Generator! This innovative tool harnesses the power of Stable Diffusion, a cutting-edge deep learning model designed to create stunning images from textual prompts.")

    prompt = st.text_input("Enter your prompt:")
    num_images = st.slider("Number of images:", 1, 5, 1)
    art_style = st.selectbox("Art Style:", ART_STYLES)
    image_size = st.selectbox("Image Size:", list(IMAGE_SIZES.keys()))
    resolution = st.selectbox("Image Resolution:", list(RESOLUTIONS.keys()))

    if st.button("Generate Image(s)"):
        if not is_valid_prompt(prompt):
            st.error("Please enter a valid prompt with at least 3 words.")
        elif is_nsfw(prompt):
            st.error("Please enter a proper prompt without NSFW content.")
        else:
            optimized_prompt, negative_prompt = optimize_prompt(prompt, art_style)
            base_width, base_height = IMAGE_SIZES[image_size]
            resolution_multiplier = RESOLUTIONS[resolution]
            width = int(base_width * resolution_multiplier)
            height = int(base_height * resolution_multiplier)

            with st.spinner("Generating images..."):
                images = []
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

                st.subheader("Generated Image(s):")
                cols = st.columns(num_images)
                for i, image in enumerate(images):
                    cols[i].image(image, use_column_width=True)
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    cols[i].download_button(
                        label="Save Image",
                        data=buf.getvalue(),
                        file_name=f"generated_image_{i+1}.png",
                        mime="image/png"
                    )

def image_to_image():
    st.header("Image to Image")
    st.write("Transform your images using AI! Upload an image and provide a prompt to guide the transformation.")

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    prompt = st.text_input("Enter your prompt:")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Generate Image"):
            if not prompt.strip():
                st.error("Please enter a prompt!")
            elif not is_valid_prompt(prompt):
                st.error("Please enter a valid prompt with at least 3 words.")
            elif is_nsfw(prompt):
                st.error("Please enter a proper prompt without NSFW content.")
            else:
                optimized_prompt, negative_prompt = optimize_prompt(prompt, "")
                init_image = image.resize((512, 512))

                with st.spinner("Generating image..."):
                    image = img2img_pipe(
                        prompt=optimized_prompt,
                        negative_prompt=negative_prompt,
                        image=init_image,
                        strength=0.75,
                        guidance_scale=7.5
                    ).images[0]

                    st.subheader("Generated Image:")
                    st.image(image, use_column_width=True)
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    st.download_button(
                        label="Save Image",
                        data=buf.getvalue(),
                        file_name="generated_image.png",
                        mime="image/png"
                    )

def image_to_text():
    st.header("Image to Text")
    st.write("Generate captions or descriptions for your images using AI!")

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Generate Caption"):
            with st.spinner("Generating caption..."):
                pixel_values = image_processor(image, return_tensors="pt").pixel_values.to("cuda")
                output_ids = image_to_text_model.generate(pixel_values, max_length=16, num_beams=4)
                caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

                st.subheader("Generated Caption:")
                st.write(caption)

if __name__ == "__main__":
    main()
