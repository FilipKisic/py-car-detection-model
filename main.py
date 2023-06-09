import os
import json
import shutil
import random
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam


def clean_work_directories(train_dir, val_dir):
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)


def load_files(train_dir, val_dir):
    annotations_file = 'cars/result.json'

    with open(annotations_file, 'r') as f:
        annotations = json.load(f)

    clean_work_directories(train_dir, val_dir)

    images = annotations['images']
    random.shuffle(images)
    return images, annotations


def create_train_class_subdirectories(annotations):
    for category in annotations['categories']:
        class_name = category['name']
        class_dir_train = os.path.join(train_dir, class_name)
        os.makedirs(class_dir_train, exist_ok=True)
        class_dirs_train[category['id']] = class_dir_train

    for image in train_images:
        src_path = os.path.join(data_dir, image['file_name'])
        if os.path.exists(src_path):
            annotation_id = image['id']
            annotation = next((ann for ann in annotations['annotations'] if ann['image_id'] == annotation_id), None)
            if annotation is not None:
                class_id = annotation['category_id']
                dst_path = os.path.join(class_dirs_train[class_id], os.path.basename(image['file_name']))
                shutil.copyfile(src_path, dst_path)


def create_val_class_subdirectories(annotations):
    for category in annotations['categories']:
        class_name = category['name']
        class_dir_val = os.path.join(val_dir, class_name)
        os.makedirs(class_dir_val, exist_ok=True)
        class_dirs_val[category['id']] = class_dir_val

    for image in val_images:
        src_path = os.path.join(data_dir, image['file_name'])
        if os.path.exists(src_path):
            annotation_id = image['id']
            annotation = next((ann for ann in annotations['annotations'] if ann['image_id'] == annotation_id), None)
            if annotation is not None:
                class_id = annotation['category_id']
                dst_path = os.path.join(class_dirs_val[class_id], os.path.basename(image['file_name']))
                shutil.copyfile(src_path, dst_path)


def define_model_architecture():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(annotations['categories']), activation='softmax'))
    return model


def setup_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    return train_datagen, val_datagen


def learning_rate_schedule(epoch):
    if epoch < 5:
        return learning_rate
    else:
        return learning_rate * tf.math.exp(-0.1)


def train_model(train_datagen, val_datagen):
    lr_scheduler = LearningRateScheduler(learning_rate_schedule)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=20,  # Increase the number of epochs
        validation_data=val_generator,
        validation_steps=val_generator.samples // val_generator.batch_size,
        callbacks=[checkpoint, lr_scheduler]
    )


def calculate_accuracy(val_dir, val_datagen):
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )

    y_true = val_generator.classes
    y_pred = model.predict(val_generator)
    y_pred = tf.argmax(y_pred, axis=1).numpy()

    accuracy = sum(y_true == y_pred) / len(y_true)
    print('Validation Accuracy:', accuracy)


# !!!!!!!!!!!!!!!!!!!!!!!!!!MAIN!!!!!!!!!!!!!!!!!!!!!!!!!!
data_dir = 'cars'
train_dir = 'train'
val_dir = 'val'

result = load_files(train_dir, val_dir)
images = result[0]
annotations = result[1]

# Split ratio for training and validation
train_ratio = 0.8
train_size = int(len(images) * train_ratio)

train_images = images[:train_size]
val_images = images[train_size:]

class_dirs_train = {}
create_train_class_subdirectories(annotations)

class_dirs_val = {}
create_val_class_subdirectories(annotations)

model = define_model_architecture()

data_generators = setup_data_generators()
train_datagen = data_generators[0]
val_datagen = data_generators[1]

# Compile the model with adjusted learning rate
learning_rate = 0.0001
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Set up model checkpoint for saving the best model
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

train_model(train_datagen, val_datagen)
calculate_accuracy(val_dir, val_datagen)

model.save('trained_model.h5')
