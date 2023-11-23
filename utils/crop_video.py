from moviepy.editor import VideoFileClip


def crop_video(input_video_path, output_video_path, start_time, end_time):
    """
    Crop a video from start_time to end_time.

    Parameters:
    - input_video_path: Path to the input video.
    - output_video_path: Path to save the cropped video.
    - start_time: Start time in format 'hh:mm:ss'.
    - end_time: End time in format 'hh:mm:ss'.
    """

    with VideoFileClip(input_video_path) as video:
        new_video = video.subclip(start_time, end_time)
        new_video.write_videofile(output_video_path, codec="libx264")


# Example usage
input_path = r'data/latest2.mp4'
output_path = r'data/short.mp4'
start = '00:00:22'  # Start at 10 seconds
end = '00:00:25'    # End at 1 minute
crop_video(input_path, output_path, start, end)
