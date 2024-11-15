import cv2
import os

def assemble_depth_video(input_folder, output_video_path, frame_rate=30):
    frame_array = []
    files = [f for f in os.listdir(input_folder) if f.endswith(".png")]
    files.sort()

    for i in range(len(files)):
        filename = os.path.join(input_folder, files[i])
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        frame_array.append(img)

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, size)

    for i in range(len(frame_array)):
        out.write(frame_array[i])

    out.release()
    print(f"Vidéo de profondeur assemblée : {output_video_path}")

if __name__ == "__main__":
    input_folder = "depth_maps"
    output_video_path = "depth_video.mp4"
    assemble_depth_video(input_folder, output_video_path)
