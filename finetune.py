import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import os

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 4  

train_dataset = keras.utils.image_dataset_from_directory(
    "Dataset_split/train",  
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

val_dataset = keras.utils.image_dataset_from_directory(
    "Dataset_split/test",  
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

normalization_layer = layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))


base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),  
    layers.Dense(NUM_CLASSES, activation="softmax")  
])

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.9, staircase=True
)

optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)  
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

base_model.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),  
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_finetune = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS // 2  
)

model.save("mobilenetv2_finetuned.h5") 
