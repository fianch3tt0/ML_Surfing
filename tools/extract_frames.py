import cv2
import os

video_folder = "videos"
output_folder = "frames"
os.makedirs(output_folder, exist_ok=True)

# extract every Nth frame (e.g., one every 8 frames)
FRAME_INTERVAL = 8

for video_file in os.listdir(video_folder):
    if not video_file.lower().endswith((".mp4", ".mov", ".m4v")):
        continue

    video_path = os.path.join(video_folder, video_file)
    vidcap = cv2.VideoCapture(video_path)

    success, frame = vidcap.read()
    frame_count = 0
    saved_count = 0

    print(f"Processing {video_file}...")

    while success:
        if frame_count % FRAME_INTERVAL == 0:
            out_path = os.path.join(
                output_folder,
                f"{os.path.splitext(video_file)[0]}_frame_{saved_count:05}.jpg",
            )
            cv2.imwrite(out_path, frame)
            saved_count += 1

        success, frame = vidcap.read()
        frame_count += 1

    print(f"Extracted {saved_count} frames from {video_file}")
