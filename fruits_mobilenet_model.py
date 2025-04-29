import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
import os
import numpy as np
import matplotlib.pyplot as plt

train_path = r"F:\Tejaswini\8th_Sem\Project_Disease_Detection\train"
test_path = r"F:\Tejaswini\8th_Sem\Project_Disease_Detection\test"

IMG_SIZE = (224, 224)  
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2 
)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

NUM_CLASSES = len(train_generator.class_indices)
print("Detected Classes:", train_generator.class_indices)
print("Number of Classes:", NUM_CLASSES)

base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

EPOCHS = 25
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    verbose=1
)

model.save("mobilenetv2_citrus_classifier.h5")
print("🎉 Model training complete and saved successfully!")

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

test_loss, test_acc = model.evaluate(test_generator, verbose=1)
print(f"✅ Test Accuracy: {test_acc * 100:.2f}%")


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

def preprocess_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_disease(model, img_path):
    processed_image = preprocess_image(img_path)
    prediction = model.predict(processed_image)
    class_names = list(train_generator.class_indices.keys())
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    return predicted_class, confidence

sample_test_images = []
for class_name in os.listdir(test_path):
    class_path = os.path.join(test_path, class_name)
    images = os.listdir(class_path)
    if images:
        sample_image = os.path.join(class_path, images[0])
        sample_test_images.append(sample_image)

plt.figure(figsize=(15, 5 * ((len(sample_test_images) + 2) // 3)))
for i, img_path in enumerate(sample_test_images):
    predicted_class, confidence = predict_disease(model, img_path)
    
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    plt.subplot(((len(sample_test_images) + 2) // 3), 3, i+1)
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}%")
    plt.axis('off')

plt.tight_layout()
plt.show()

print("\nPredictions for Sample Test Images:")
for img_path in sample_test_images:
    predicted_class, confidence = predict_disease(model, img_path)
    true_class = os.path.basename(os.path.dirname(img_path))
    print(f"Image: {os.path.basename(img_path)}")
    print(f"True Class: {true_class}")
    print(f"Predicted: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    print()

print("Prediction complete!")









