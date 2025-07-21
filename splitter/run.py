from dotenv import load_dotenv
from pathlib import Path
import os
import splitter


load_dotenv()
VIDEO_INPUT = os.getenv('VIDEO_INPUT')


if __name__ == "__main__":
    input_path = Path(VIDEO_INPUT)
    file_name = input_path.stem
    extension = input_path.suffix
    output_path_frames = input_path.with_name(f"frames {file_name}")
    output_path_audio = input_path.with_name(f"result {file_name}.mp3")
    output_path_result = input_path.with_name(f"result {file_name}{extension}")
    splitter.split(VIDEO_INPUT, output_path_frames)
    splitter.extract_audio(VIDEO_INPUT, output_path_audio)
    fps = splitter.get_fps(VIDEO_INPUT)
    splitter.join(output_path_frames, output_path_result, fps, audio_path=output_path_audio)
