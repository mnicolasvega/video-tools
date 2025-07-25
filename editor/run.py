from dotenv import load_dotenv
from pathlib import Path
import editor
import os

load_dotenv()
VIDEO_INPUT = os.getenv('VIDEO_INPUT')

def get_output_path(input: str, label: str) -> str:
    file_path = Path(input)
    file_name = file_path.stem
    extension = file_path.suffix
    return file_path.with_name(f"{file_name} {label}{extension}")

if __name__ == "__main__":
    output_path = get_output_path(VIDEO_INPUT, 'edit')
    duration = editor.get_duration(VIDEO_INPUT)
    bounds = editor.edit(
        VIDEO_INPUT,
        output_path,
        second_start = duration - 3 * 60,
        output_width = 640,
        output_height = 360
    )
