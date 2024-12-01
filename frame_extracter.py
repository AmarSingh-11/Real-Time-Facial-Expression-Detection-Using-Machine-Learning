import cv2
import os

def extract_frames(video_path, output_folder, frame_interval=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Create output folder if it doesn't exist
    
    cap = cv2.VideoCapture(video_path)
    count = 0
    success = True
    video_name = os.path.basename(video_path).split('.')[0]  # Get video name without extension

    while success:
        success, image = cap.read()
        if not success:
            break  # Exit the loop if no more frames to read
        
        if count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"{video_name}_frame_{count}.jpg")
            cv2.imwrite(frame_filename, image)
        
        count += 1
    
    cap.release()

# Folder containing all the videos
video_folder = "/home/amar-singh/Desktop/Machine learning/Random_forest/neutral_video/Aditya/Neutral"

# Output folder where frames will be saved
output_folder = "/home/amar-singh/Desktop/Machine learning/Random_forest/neutral_video/extracted_frames_a_neutral"

# Get all video files in the directory with .avi extension
video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]

# Loop through all video files and extract frames
for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)
    extract_frames(video_path, output_folder=output_folder)

