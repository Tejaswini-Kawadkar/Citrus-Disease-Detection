import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Constants
IMG_SIZE = (224, 224)
MODEL_PATH = "mobilenetv2_citrus_leaves_classifier.h5"
TEST_PATH = r"F:\Tejaswini\8th_Sem\Citrus_Leaves" 
NUM_PREDICTIONS = 10

# Load the model
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

# Manually define class labels by reading the directories in TEST_PATH
class_labels = sorted([folder for folder in os.listdir(TEST_PATH) if os.path.isdir(os.path.join(TEST_PATH, folder))])

def preprocess_image(img_path):
    """Preprocess the image for prediction."""
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array

def predict_disease(img_path):
    """Predict the disease class and confidence for the given image."""
    processed_image = preprocess_image(img_path)
    prediction = model.predict(processed_image, verbose=0)  # Suppress verbose output
    predicted_class = class_labels[np.argmax(prediction)]  # Use the class label
    confidence = np.max(prediction) * 100
    return predicted_class, confidence

def get_random_test_images(num_images=10):
    image_paths = []
    class_folders = [os.path.join(TEST_PATH, folder) for folder in os.listdir(TEST_PATH) if os.path.isdir(os.path.join(TEST_PATH, folder))]
    
    # Adjust the number of images per class to ensure the total is num_images
    num_classes = len(class_folders)
    images_per_class = num_images // num_classes
    extra_images = num_images % num_classes  # To distribute remaining images

    for i, class_folder in enumerate(class_folders):
        class_images = [os.path.join(class_folder, img) for img in os.listdir(class_folder) if img.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Select images for the current class
        selected_images = random.sample(
            class_images, 
            min(len(class_images), images_per_class + (1 if i < extra_images else 0))  # Distribute extra images evenly
        )
        image_paths.extend(selected_images)
    
    # If fewer than num_images, randomly sample from the available images
    if len(image_paths) < num_images:
        print(f"Warning: Only {len(image_paths)} images found. Desired number is {num_images}.")
    
    return image_paths

def plot_predictions(test_images):
    """Plot the predictions with their true and predicted labels."""
    plt.figure(figsize=(20, 20))
    for i, img_path in enumerate(test_images):
        true_class = os.path.basename(os.path.dirname(img_path))
        predicted_class, confidence = predict_disease(img_path)
        
        img = image.load_img(img_path, target_size=IMG_SIZE)
        plt.subplot(4, 3, i + 1)
        plt.imshow(img)
        plt.title(
            f"\n\n\n\n\n\nActual: {true_class}\nPredicted: {predicted_class}\nConfidence: {confidence:.2f}%",
            fontsize=10
        )
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def evaluate_model():
    """Evaluate the model on the test set."""
    y_true = []
    y_pred = []
    class_labels = sorted([folder for folder in os.listdir(TEST_PATH) if os.path.isdir(os.path.join(TEST_PATH, folder))])

    for class_name in os.listdir(TEST_PATH):
        class_path = os.path.join(TEST_PATH, class_name)
        if os.path.isdir(class_path):
            images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith(('.jpg', '.jpeg', '.png'))]
            for img_path in images:
                true_class = class_name.lower()  # Ensure true class is in lower case
                predicted_class, _ = predict_disease(img_path)
                y_true.append(true_class)
                y_pred.append(predicted_class.lower())  # Ensure predicted class is in lower case

    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))

    # Plot confusion matrix
    unique_labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='PuBu', xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    

# Main Execution
test_images = get_random_test_images(NUM_PREDICTIONS)
print(f"Randomly selected {len(test_images)} images for predictions.")

# Plot predictions
plot_predictions(test_images)

# Print individual predictions
print("\nPredictions for Test Images:")
for i, img_path in enumerate(test_images):
    predicted_class, confidence = predict_disease(img_path)
    true_class = os.path.basename(os.path.dirname(img_path))
    print(f"{i + 1}. Image: {os.path.basename(img_path)}")
    print(f"   True Class: {true_class}")
    print(f"   Predicted: {predicted_class}")
    print(f"   Confidence: {confidence:.2f}%")
    print()

# Evaluate model
evaluate_model()
print("Evaluation complete!")

