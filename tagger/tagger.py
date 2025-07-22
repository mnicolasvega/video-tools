from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

MODEL = "Salesforce/blip-image-captioning-base"

def run(img_path: str) -> str:
    raw_image = Image.open(img_path).convert('RGB')
    processor = BlipProcessor.from_pretrained(MODEL)
    model = BlipForConditionalGeneration.from_pretrained(MODEL)

    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption
