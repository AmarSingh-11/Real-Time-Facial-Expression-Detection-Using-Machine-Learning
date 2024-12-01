import cv2
import os
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# Step 2: Load images and extract features
base_dir = "/home/amar-singh/Desktop/Machine learning/balanced/grayscale_new/balanced"  # Change to your grayscale images directory
classes = ["sad", "neutral", "happy"]

data = []
labels = []

# Load images and labels
for label, emotion in enumerate(classes):
    emotion_dir = os.path.join(base_dir, emotion)
    
    for img_name in os.listdir(emotion_dir):
        img_path = os.path.join(emotion_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize if not already done
        img = cv2.resize(img, (48, 48))
        
        # Append image and label
        data.append(img)
        labels.append(label)

# Convert to numpy arrays
X = np.array(data)
y = np.array(labels)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# HOG feature extraction
def extract_hog_features(images):
    hog_features = []
    for img in images:
        features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        hog_features.append(features)
    return np.array(hog_features)

# LBP feature extraction
def extract_lbp_features(images):
    lbp_features = []
    for img in images:
        lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
        lbp_features.append(hist)
    return np.array(lbp_features)

# Extract features
X_train_hog = extract_hog_features(X_train)
X_test_hog = extract_hog_features(X_test)
X_train_lbp = extract_lbp_features(X_train)
X_test_lbp = extract_lbp_features(X_test)

# Combine HOG and LBP features
X_train_combined = np.hstack((X_train_hog, X_train_lbp))
X_test_combined = np.hstack((X_test_hog, X_test_lbp))

# Step 3: Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_combined, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test_combined)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=classes))

