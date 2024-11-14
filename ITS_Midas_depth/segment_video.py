import shutil
import cv2
import os

def clear_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def segment_video(video_path, output_folder, frame_skip=1):
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else :
        clear_folder(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            frame_path = os.path.join(output_folder, f"frame_{saved_frame_count:04d}.png")
            cv2.imwrite(frame_path, frame)
            saved_frame_count += 1
        frame_count += 1

    cap.release()
    print(f"Vidéo segmentée en {saved_frame_count} images dans {output_folder}")

if __name__ == "__main__":
    video_path = "public_domain.mp4"
    output_folder = "frames"
    frame_skip = 30  # 1 out of 30 frames otherwise it takes too long
    segment_video(video_path, output_folder, frame_skip)
