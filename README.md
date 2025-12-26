# Image Caption Generator using Pretrained BLIP Model

This project generates descriptive captions for images using a
pretrained BLIP (Bootstrapped Language Image Pretraining) model.
The focus of this project is on understanding and implementing
the inference pipeline for image captioning.

## Technologies Used
- Python
- Transformers (Hugging Face)
- PyTorch
- BLIP Pretrained Model
- Google Colab

## Project Description
Training large vision-language models from scratch requires
massive datasets and high computational resources.
In this project, a pretrained BLIP model is used to explore
how image captioning works in practice.

The implementation allows users to upload an image and
automatically generate a meaningful caption for it.

## Files
- caption_generator_colab.py  
  Script to generate image captions using a pretrained BLIP model
  in a Google Colab environment.

## Working Flow
1. User uploads an image.
2. The image is processed using the BLIP processor.
3. The pretrained BLIP model generates a caption.
4. The image and generated caption are displayed as output.

## Output
- Uploaded image
- Generated textual caption describing the image

## Applications
- Assistive technology for visually impaired users
- Automatic image tagging
- Content generation systems

## Notes
- This project uses a pretrained model for inference only.
- Internet connection is required to download the model weights.
- GPU is optional but improves performance.

## Future Improvements
- Support batch image captioning
- Experiment with different pretrained models
- Build a simple web interface
