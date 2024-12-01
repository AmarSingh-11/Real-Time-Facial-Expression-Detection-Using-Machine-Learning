import os
import cv2
from deepface import DeepFace

# Define directories
input_dir = '/home/amar-singh/Desktop/Machine learning/Random_forest/neutral_video/extracted_faces'
output_dir = "/home/amar-singh/Desktop/Machine learning/Random_forest/neutral_video/neutral"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

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
        
        # Check if the result contains the expected fields
        if result:
            emotion = result[0]['dominant_emotion']
            score = result[0]['emotion'][emotion]

            # Filter for "sad" emotion with score >= 80
            if emotion == "happy" and score >= 85:
                print(f"Detected emotion: {emotion} with score: {score} for {filename}")

                # Save the image to the "Sad" directory
                cv2.imwrite(os.path.join(output_dir, filename), image)

    except Exception as e:
        print(f"Error processing {filename}: {e}")

