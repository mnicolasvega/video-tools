from moviepy import VideoFileClip, concatenate_videoclips

CODEC_VIDEO = "libx264"
CODEC_AUDIO = "aac"

def get_duration(input_path: str) -> int:
    return VideoFileClip(input_path).duration

def edit(
        input_path: str,
        output_path: str | None = None,
        second_start: int | None = None,
        second_end: int | None = None,
        output_width: int | None = None,
        output_height: int | None = None,
    ) -> VideoFileClip:
    video = VideoFileClip(input_path)

    if second_start == None:
        second_start = 0
    if second_end == None:
        second_end = video.duration
    if output_width == None:
        output_width = video.w
    if output_height == None:
        output_height = video.h

    duration_start = max(0, second_start)
    duration_end = min(second_end, video.duration)
    video = video.subclipped(duration_start, duration_end)
    video = video.resized(new_size = (output_width, output_height))

    if output_path:
        video.write_videofile(output_path, codec = CODEC_VIDEO, audio_codec = CODEC_AUDIO)

    return video

def merge(video_paths: list, output_path: str) -> None:
    clips = [VideoFileClip(path) for path in video_paths]
    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    for clip in clips:
        clip.close()
    final_clip.close()
