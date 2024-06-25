import os
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import random
import datetime 

# Decision keyword
decision = 'train'  # Use 'train' to train a new model, 'use' to use an existing model

# Paths to data directories
base_dir = "C:/Users/jack/Coding Shit/Stock Project/Data/Graph Data/"
model_save_path = "C:/Users/jack/Coding Shit/Stock Project/Models/stock_performance_cnn.h5"
increase_dir = os.path.join(base_dir, 'Week20')  # Adjust model type as needed
underperform_dir = os.path.join(base_dir, 'Week-Underperform')

# Image parameters
img_width, img_height = 640, 480
batch_size = 32
max_samples = 500  # Set a limit for the number of images per class

# Function to load images and labels from a directory
def load_images_from_directory(directory, label, max_samples=None):
    images = []
    labels = []
    file_list = os.listdir(directory)
    if max_samples is not None:
        file_list = random.sample(file_list, min(len(file_list), max_samples))
    for fname in file_list:
        img_path = os.path.join(directory, fname)
        if img_path.lower().endswith(('png', 'jpg', 'jpeg')):
            img = load_img(img_path, target_size=(img_width, img_height))
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(label)
    return images, labels

if decision == 'train':
    # Load images and labels with a limit on the number of samples
    increase_images, increase_labels = load_images_from_directory(increase_dir, 1, max_samples)
    underperform_images, underperform_labels = load_images_from_directory(underperform_dir, 0, max_samples)

    total_images = len(increase_images) + len(underperform_images)
    increase_percentage, underperform_percentage = round(len(increase_images)/total_images*100,3), round(len(underperform_images)/total_images,3)

    print('Overperform Distribution: ' + str(increase_percentage) + '  ||  Underperform Distribution: ' + str(underperform_percentage))

    # Combine the data and split into training and validation sets
    X = np.array(increase_images + underperform_images)
    y = np.array(increase_labels + underperform_labels)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Data augmentation
    train_datagen = ImageDataGenerator(rescale=1.0/255, horizontal_flip=True, zoom_range=0.2)
    val_datagen = ImageDataGenerator(rescale=1.0/255)

    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    validation_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

    # Define the CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),

        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    epochs = 30
    steps_per_epoch = len(X_train) // batch_size
    validation_steps = len(X_val) // batch_size

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps
    )

    # Save the model
    model.save(model_save_path)

    # Evaluate the model
    validation_loss, validation_accuracy = model.evaluate(validation_generator, steps=validation_steps)
    print(f'Validation accuracy: {validation_accuracy*100:.2f}%')

else:
    # Load the existing model
    model = load_model(model_save_path)

# In-practice data directory
in_practice_dir = os.path.join(base_dir, 'In-Practice')

# Function to make predictions and save to CSV
def make_predictions_and_save(model, directory, output_csv):
    images = []
    file_names = []
    for fname in os.listdir(directory):
        img_path = os.path.join(directory, fname)
        if img_path.lower().endswith(('png', 'jpg', 'jpeg')):
            img = load_img(img_path, target_size=(img_width, img_height))
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            file_names.append(fname)

    images = np.array(images)
    predictions = model.predict(images)
    predictions = (predictions > 0.5).astype(int)

    df = pd.DataFrame({'File Name': file_names, 'Prediction': predictions.flatten()})
    df.to_csv(output_csv, index=False)

# Make predictions and save to CSV
output_csv = "C:/Users/jack/Coding Shit/Stock Project/Predictions/StockCNN-Preds-" + str(datetime.datetime.today() + ".csv"
make_predictions_and_save(model, in_practice_dir, output_csv)
print(f'Predictions saved to {output_csv}')
