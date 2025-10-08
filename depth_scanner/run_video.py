from dotenv import load_dotenv
from pathlib import Path
from torchvision.transforms import Compose
import depth
import os
import sys


load_dotenv()
PATH_FOLDER_INPUT = os.getenv('PATH_FOLDER_INPUT')
PATH_FOLDER_OUTPUT = os.getenv('PATH_FOLDER_OUTPUT')
MIDAS_PATH = os.getenv('MIDAS_PATH')
SAVE_FORMAT = "png"
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

sys.path.append(MIDAS_PATH)
from midas.transforms import NormalizeImage, PrepareForNet, Resize

def get_output_path(input_path: Path) -> Path:
    output_folder = Path(PATH_FOLDER_OUTPUT)
    output_path = output_folder / f"{input_path.name}"
    return output_path

def get_custom_transform(size: int = 384):
    """Get custom MiDaS transform similar to parse_video_to_vr.py"""
    return Compose([
        Resize(size, size, resize_target=False, keep_aspect_ratio=True),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet()
    ])

def process_folder(folder_path: str):
    folder = Path(folder_path)
    output_folder = Path(PATH_FOLDER_OUTPUT)
    
    if not folder.exists() or not folder.is_dir():
        print(f"Error: {folder_path} is not a valid directory")
        return
    
    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = sorted([f for f in folder.iterdir() 
                          if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS])
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    model, device, _ = depth.get_midas()
    transform = get_custom_transform()
    for idx, image_path in enumerate(image_files, 1):
        print(f"Processing {idx}/{len(image_files)}: {image_path.name}")
        try:
            output_path = get_output_path(image_path)
            if output_path.exists():
                #print(f"  Skipping (already exists): {output_path}")
                continue
            depth_map = depth.run_with_custom_transform(str(image_path), model, device, transform)
            image = depth.to_image(depth_map)
            image.save(str(output_path), format=SAVE_FORMAT)
            print(f"  Saved to: {output_path}")
        except Exception as e:
            print(f"  Error processing {image_path.name}: {e}")
    
    print(f"\nProcessing complete! Output saved to: {output_folder}")

if __name__ == "__main__":
    if not PATH_FOLDER_INPUT:
        print("Error: PATH_FOLDER_INPUT not set in .env file")
    elif not PATH_FOLDER_OUTPUT:
        print("Error: PATH_FOLDER_OUTPUT not set in .env file")
    else:
        process_folder(PATH_FOLDER_INPUT)
