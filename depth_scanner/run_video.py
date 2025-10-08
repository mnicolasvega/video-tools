from dotenv import load_dotenv
from pathlib import Path
import depth
import os

load_dotenv()
PATH_FOLDER_INPUT = os.getenv('PATH_FOLDER_INPUT')
SAVE_FORMAT = "jpg"
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

def get_output_path(input_path: Path, output_folder: Path) -> Path:
    output_path = output_folder / f"{input_path.stem} depth.{SAVE_FORMAT}"
    return output_path

def process_folder(folder_path: str):
    folder = Path(folder_path)
    
    if not folder.exists() or not folder.is_dir():
        print(f"Error: {folder_path} is not a valid directory")
        return
    
    # Create output folder
    output_folder = folder / "depth_output"
    output_folder.mkdir(exist_ok=True)
    
    # Get all image files
    image_files = [f for f in folder.iterdir() 
                   if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for idx, image_path in enumerate(image_files, 1):
        print(f"Processing {idx}/{len(image_files)}: {image_path.name}")
        try:
            output_path = get_output_path(image_path, output_folder)
            depth_map = depth.run(str(image_path))
            image = depth.to_image(depth_map)
            image.save(output_path, format=SAVE_FORMAT)
            print(f"  Saved to: {output_path}")
        except Exception as e:
            print(f"  Error processing {image_path.name}: {e}")
    
    print(f"\nProcessing complete! Output saved to: {output_folder}")

if __name__ == "__main__":
    if not PATH_FOLDER_INPUT:
        print("Error: PATH_FOLDER_INPUT not set in .env file")
    else:
        process_folder(PATH_FOLDER_INPUT)
