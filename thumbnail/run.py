from dotenv import load_dotenv
from pathlib import Path
import os
import thumbnail

load_dotenv()
VIDEO_INPUT = os.getenv('VIDEO_INPUT')
CODEC_VIDEO = "libx264"
CODEC_AUDIO = "aac"

def get_output_path(input_path: str) -> str:
    file_path = Path(input_path)
    file_name = file_path.stem
    return file_path.with_name(f"{file_name}")

if __name__ == "__main__":
    output_path = get_output_path(VIDEO_INPUT)
    video_name = Path(VIDEO_INPUT).stem
    output_clips = f"{video_name}/clips"
    output_thumbnails = f"{video_name}/thumbnails"
    os.makedirs(output_clips, exist_ok=True)
    os.makedirs(output_thumbnails, exist_ok=True)
    scenes = thumbnail.detect_scenes(VIDEO_INPUT)
    
    print("getting thumbnails")
    thumbnails = thumbnail.get_thumbnails(VIDEO_INPUT, scenes)
    print("saving thumbnails")
    i = 0
    for timestamp, thumbnail_clip in thumbnails.items():
        thumbnail_path = f"{output_thumbnails}/scene_{i}_{timestamp}.jpg"
        thumbnail_clip.save(thumbnail_path)
        i = i + 1

    print("creating clips")
    clips = thumbnail.get_clips(VIDEO_INPUT, scenes)
    print("saving clips")
    for timestamp, clip in clips.items():
        clip_path = f"{output_clips}/scene_{timestamp}.mp4"
        clip = clip.resized(new_size = (640, 360))
        clip.write_videofile(clip_path, codec = CODEC_VIDEO, audio_codec = CODEC_AUDIO)
