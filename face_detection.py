import dlib
import cv2
import os

# Initialize face detector
detector = dlib.get_frontal_face_detector()

def detect_faces(frame_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for frame_file in os.listdir(frame_folder):
        frame_path = os.path.join(frame_folder, frame_file)
        image = cv2.imread(frame_path)
        
        if image is None:
            print(f"Warning: Unable to read image {frame_path}")
            continue  # Skip this file if it cannot be read
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = detector(gray, 1)
        
        if len(faces) == 0:
            print(f"No faces detected in {frame_file}")
            continue  # Skip to the next frame if no faces are found
        
        for i, rect in enumerate(faces):
            x, y, w, h = rect.left(), rect.top(), rect.right(), rect.bottom()
            # Crop and save detected face
            face_crop = image[y:h, x:w]
            
            if face_crop.size == 0:
                print(f"Warning: Detected face region is empty in {frame_file}")
                continue  # Skip if the cropped image is empty
            
            face_filename = f"{output_folder}/{frame_file.split('.')[0]}_face_{i}.jpg"
            cv2.imwrite(face_filename, face_crop)

# Detect faces in the extracted frames
frames_folder = "/home/amar-singh/Desktop/Machine learning/Random_forest/neutral_video/extracted_frames_a_neutral"
faces_output_folder = "/home/amar-singh/Desktop/Machine learning/Random_forest/neutral_video/extracted_faces_neutral_a"
detect_faces(frames_folder, faces_output_folder)

