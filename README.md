# DreamCanvas

DreamCanvas is an AI-powered web application that transforms text and images into visual creations using state-of-the-art deep learning models for text-to-image generation, image-to-image transformation, and image captioning.

## Table of Contents
1. [Features](#features)
2. [Technical Architecture](#technical-architecture)
3. [Deployment Instructions](#deployment-instructions)
4. [Challenges Faced and Solutions](#challenges-faced-and-solutions)
5. [Evaluation Criteria](#evaluation-criteria)
6. [Additional Notes](#additional-notes)

## Features

1. **Text to Image**: Generate up to 5 images from text prompts, with customizable art styles, image sizes, and resolutions.
2. **Image to Image**: Upload an image and transform it based on a text prompt.
3. **Image to Text**: Generate captions for uploaded images using AI-powered image captioning.

## Technical Architecture

- **Backend**: Streamlit (Python)
- **Frontend**: HTML, CSS (integrated within Streamlit)
- **AI Models**:
  - Stable Diffusion v1-5 for text-to-image and image-to-image generation.
  - VisionEncoderDecoderModel (ViT-GPT2) for image captioning.
- **Key Libraries**: PyTorch, Hugging Face Transformers, Diffusers, PIL, and Streamlit.

### Model Integration
- **Text-to-Image**: Uses the `StableDiffusionPipeline` for generating images from text prompts. It allows customization of art styles, image resolutions, and sizes.
- **Image-to-Image**: Uses the `StableDiffusionImg2ImgPipeline` to transform uploaded images with textual guidance.
- **Image Captioning**: Utilizes a pre-trained `VisionEncoderDecoderModel` to generate captions for uploaded images.

### Optimization Techniques
- **Memory Optimization**: Attention slicing is enabled for Stable Diffusion models to reduce memory usage.
- **GPU Utilization**: Models are run on GPU (using CUDA) for faster inference.
- **Prompt Optimization**: Prompts are optimized with additional descriptors to generate high-quality images.
- **NSFW Content Filtering**: A basic filter is implemented to detect and block inappropriate content in prompts.

## Deployment Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/dreamcanvas.git
   cd dreamcanvas
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

5. Open your web browser and navigate to `http://localhost:8501`.

### Scaling Considerations
- Implement a job queue system (e.g., Redis Queue) for handling multiple simultaneous requests.
- Use a load balancer to distribute traffic across multiple servers.
- Consider implementing caching for frequently generated images or captions.

## Challenges Faced and Solutions

1. **Memory Management**: 
   - Challenge: The models require significant GPU memory.
   - Solution: Implemented attention slicing and moved models to GPU for efficient processing.

2. **Input Validation**: 
   - Challenge: Ensuring the input prompts are safe and free from inappropriate content.
   - Solution: Regex-based validation and NSFW content filtering were added to check prompt inputs.

3. **Performance Optimization**: 
   - Challenge: Generating high-quality images can take time.
   - Solution: Optimized prompt formatting, model settings, and enabled GPU-based inference for faster results.

4. **Handling Large Models**: 
   - Challenge: Stable Diffusion models are resource-intensive.
   - Solution: Used model precision (torch.float16) and enabled model-specific optimizations like attention slicing.

## Evaluation Criteria

1. **Functionality**: 
   - The prototype successfully generates images from text prompts, transforms uploaded images, and captions images using AI.
   - Thorough testing has been conducted to ensure reliability and relevance of outputs.

2. **Technical Soundness**: 
   - The codebase follows best practices and is modular. Models are efficiently integrated with the Hugging Face Transformers and Diffusers libraries.
   - Code and logic for AI interaction are optimized for GPU use and low memory footprint.

3. **Creativity**: 
   - Features multiple art styles, image size options, and resolutions for creative image generation.
   - Input validation and prompt optimization techniques enhance the quality of outputs.

4. **Deployment**: 
   - The application is ready for deployment on cloud platforms with minimal configuration.
   - Instructions for local setup and scalability considerations are provided.

5. **Problem-solving**: 
   - Resourceful solutions to GPU memory constraints and user input validation.
   - The prompt optimization method ensures high-quality results, and feedback mechanisms improve user interaction.

## Additional Notes

- For improved performance, consider fine-tuning models on domain-specific data.
- Regular updates to the underlying models and libraries are recommended to keep up with advances in AI.
- Make sure to comply with the licensing terms of the models and libraries used, especially for commercial applications.

---

We hope you enjoy using DreamCanvas! For any questions or contributions, please open an issue or submit a pull request.
