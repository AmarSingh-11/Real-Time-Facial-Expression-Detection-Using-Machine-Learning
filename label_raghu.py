import os
import cv2
from deepface import DeepFace

# Define directories
input_dir = '/home/amar-singh/Desktop/Machine learning/Random_forest/neutral_video/extracted_faces'
output_dir = "/home/amar-singh/Desktop/Machine learning/Random_forest/neutral_video/neutral"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process each image and label it
for filename in os.listdir(input_dir):
    file_path = os.path.join(input_dir, filename)
    image = cv2.imread(file_path)
    if image is None:
        print(f"Failed to read image {file_path}")
        continue

    # Analyze the image using DeepFace
    try:
        result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
        if result:
            emotion = result[0]['dominant_emotion']
            score = result[0]['emotion'][emotion]
            print(f"Detected emotion: {emotion} with score: {score} for {filename}")

            # Save the image to a directory named after the emotion
            label_dir = os.path.join(output_dir, emotion)
            os.makedirs(label_dir, exist_ok=True)
            cv2.imwrite(os.path.join(label_dir, filename), image)

    except Exception as e:
        print(f"Error processing {filename}: {e}")

