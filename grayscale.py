import cv2
import os
import numpy as np

# Directory paths
input_dir = "/home/amar-singh/Desktop/Machine learning/balanced/Human_validated"  # Directory containing balanced RGB images
output_dir = "/home/amar-singh/Desktop/Machine learning/balanced/grayscale_new" # Directory to save grayscale images

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Convert images
for emotion in ["sad", "neutral", "happy"]:
    emotion_dir = os.path.join(input_dir, emotion)
    grayscale_emotion_dir = os.path.join(output_dir, emotion)
    os.makedirs(grayscale_emotion_dir, exist_ok=True)

    for img_name in os.listdir(emotion_dir):
        # Read the image
        img_path = os.path.join(emotion_dir, img_name)
        img = cv2.imread(img_path)

        # Convert to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize to desired dimensions if necessary
        gray_img = cv2.resize(gray_img, (48, 48))

        # Save the grayscale image
        cv2.imwrite(os.path.join(grayscale_emotion_dir, img_name), gray_img)

print("Conversion to grayscale completed.")

