import pandas as pd
from pytube import YouTube


def download_highest_quality_video(video_url, output_dir):
    try:
        yt = YouTube(video_url)
        video_stream = yt.streams.filter(
            progressive=True, file_extension="mp4").order_by("resolution").desc().first()

        if video_stream:
            print(f"Downloading {yt.title} in {video_stream.resolution}...")
            video_stream.download(output_path=output_dir)
            print(f"{yt.title} downloaded successfully.")
        else:
            print(f"No video streams available for {yt.title}")
    except Exception as e:
        print(f"Error downloading video: {str(e)}")


output_directory = r"./data"
video_url = r"https://www.youtube.com/watch?v=bVz-ywcz464"

download_highest_quality_video(video_url, output_directory)
