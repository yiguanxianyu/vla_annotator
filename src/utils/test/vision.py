from torchvision import io, transforms

video_path = (
    "test_data/observations/616/843991/videos/head_color.mp4"
)

video, audio, info = io.read_video(
    video_path,
    start_pts=0.0,
    end_pts=None,
    pts_unit="sec",
    output_format="TCHW",
)

print(video.shape)
print(info)
