import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, BatchNormalization
from tensorflow.keras.models import Model

# Dataset path
dataset_path = "dataset"

# Data Generator (Important for accuracy)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

train_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

# Load Pretrained Model (CNN Backbone)
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base layers (important for stability)
for layer in base_model.layers:
    layer.trainable = False

# Safe Custom CNN Block (NO dimension crash)
x = base_model.output
x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)

# Classification Head
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(4, activation='softmax')(x)

# Final Model
model = Model(inputs=base_model.input, outputs=output)

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train (Increase epochs for better accuracy)
model.fit(
    train_data,
    validation_data=val_data,
    epochs=
)

# Save Model
model = tf.keras.models.load_model("Blood cell.h5")
print("Model Trained Successfully with CNN + Transfer Learning!")

