!pip install transformers torch pillow -q

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from google.colab import files
import torch
import matplotlib.pyplot as plt

# Load BLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

while True:
    print("\nUpload an image to generate caption:")
    uploaded = files.upload()
    if not uploaded:
        print("No file uploaded. Exiting.")
        break

    image_path = list(uploaded.keys())[0]
    img = Image.open(image_path).convert("RGB")

    inputs = processor(img, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_length=20)
    caption = processor.decode(out[0], skip_special_tokens=True)

    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Caption: {caption}")
    plt.show()

    print("Generated Caption:", caption)
