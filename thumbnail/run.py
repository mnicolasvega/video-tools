from dotenv import load_dotenv
from moviepy import concatenate_videoclips
from pathlib import Path
import os
import thumbnail

load_dotenv()
VIDEO_INPUT = os.getenv('VIDEO_INPUT')
CODEC_VIDEO = "libx264"
CODEC_AUDIO = "aac"
FORMAT_THUMBNAIL = "png"
FORMAT_VIDEO = "mp4"
CONFIG_MERGE_CLIPS = True
CONFIG_CREATE_THUMBNAILS = True
CONFIG_CREATE_CLIPS = True

def get_output_path(input_path: str) -> str:
    file_path = Path(input_path)
    file_name = file_path.stem
    return str(file_path.with_name(f"{file_name}"))

if __name__ == "__main__":
    output_path = get_output_path(VIDEO_INPUT)
    output_clips = f"{output_path}/clips"
    output_thumbnails = f"{output_path}/thumbnails"
    os.makedirs(output_clips, exist_ok=True)
    os.makedirs(output_thumbnails, exist_ok=True)
    scenes = thumbnail.detect_scenes(VIDEO_INPUT)
    clips = []

    if CONFIG_CREATE_THUMBNAILS and len(scenes) > 0:
        print(f"getting thumbnails: '{output_thumbnails}'")
        thumbnails = thumbnail.get_thumbnails(VIDEO_INPUT, scenes)
        print("saving thumbnails")
        i = 0
        for timestamp, thumbnail_clip in thumbnails.items():
            thumbnail_path = f"{output_thumbnails}/scene_{i}_{timestamp}.{FORMAT_THUMBNAIL}"
            thumbnail_clip.save(thumbnail_path)
            i = i + 1

    if CONFIG_CREATE_CLIPS and len(scenes) > 0:
        print(f"creating clips: '{output_clips}'")
        clips = thumbnail.get_clips(VIDEO_INPUT, scenes)
        print("saving clips")
        i = 0
        for timestamp, clip in clips.items():
            clip_path = f"{output_clips}/scene_{i}_{timestamp}.{FORMAT_VIDEO}"
            clip = clip.resized(new_size = (640, 360))
            clips.append(clip)
            clip.write_videofile(clip_path, codec = CODEC_VIDEO, audio_codec = CODEC_AUDIO)
            i = i + 1

    if CONFIG_MERGE_CLIPS and len(clips) > 0:
        clip_path = f"{output_path}/merge.{FORMAT_VIDEO}"
        print(f"merging clips: '{clip_path}'")
        clip_recomposed = concatenate_videoclips(clips, method="compose")
        clip_recomposed.write_videofile(clip_path, codec = CODEC_VIDEO, audio_codec = CODEC_AUDIO)
