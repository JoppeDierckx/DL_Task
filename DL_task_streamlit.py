import os
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# Function to perform EDA on the downloaded dataset
def perform_eda(dataset_path):
    # Get a list of categories in the dataset excluding the first folder
    categories = [category for category in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, category)) and category.lower() != 'folder_to_exclude']

    # Count the number of images in each category
    image_counts = {}
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        image_counts[category] = len([file for file in os.listdir(category_path) if file.endswith('.jpg')])

    # Print the number of images in each category
    st.write("Number of images in each category:")
    for category, count in image_counts.items():
        st.write(f"{category}: {count} images")

    # Visualize a few images from each category
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        image_files = [file for file in os.listdir(category_path) if file.endswith('.jpg')]

        # Display a few images from each category
        st.write(f"\nSample images from {category}:")
        for i in range(min(3, len(image_files))):  # Display up to 3 images per category
            image_path = os.path.join(category_path, image_files[i])
            img = plt.imread(image_path)
            st.image(img, caption=f"{category}_{i + 1}", use_column_width=True)

# Function to train the model
def train_model(train_generator, validation_generator, epochs):
    # Define the model
    NUM_CLASSES = len(train_generator.class_indices)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )

    return model, history

# Streamlit app
st.title("Deep Learning Model Training and EDA")

# Sidebar controls
dataset_path = st.text_input("Enter dataset path:", r"C:\Users\Joppe\Documents\school\Jaar_3\AI\11. DL - Introduction to Deep Learning")
perform_eda_button = st.button("Perform EDA")

# Perform EDA
if perform_eda_button:
    perform_eda(dataset_path)

# Train the model
train_model_button = st.button("Train Model")
if train_model_button:
    # Specify the subdirectories for train, validation, and test
    train_dir = os.path.join(dataset_path, 'train')
    validation_dir = os.path.join(dataset_path, 'validation')
    test_dir = os.path.join(dataset_path, 'test')

    # Create data generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    st.sidebar.header("Model Training Controls")
    epochs = st.sidebar.slider("Number of Epochs", min_value=1, max_value=20, value=10)
    model, history = train_model(train_generator, validation_generator, epochs)

    # Plot training and validation loss
    st.subheader("Training and Validation Loss")
    st.line_chart({
        'Training Loss': history.history['loss'],
        'Validation Loss': history.history['val_loss']
    })

    # Plot training and validation accuracy
    st.subheader("Training and Validation Accuracy")
    st.line_chart({
        'Training Accuracy': history.history['accuracy'],
        'Validation Accuracy': history.history['val_accuracy']
    })

    # Evaluate the model on the test set
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_loss, test_accuracy = model.evaluate(test_generator)
    st.subheader("Test Evaluation")
    st.write(f'Test Loss: {test_loss:.4f}')
    st.write(f'Test Accuracy: {test_accuracy:.4f}')

    # Predict classes for the test set
    predictions = model.predict(test_generator)
    predicted_classes = predictions.argmax(axis=1)

    # Get true classes for the test set
    true_classes = test_generator.classes

    # Compute confusion matrix
    conf_matrix = confusion_matrix(true_classes, predicted_classes)

    # Plot confusion matrix as a heatmap
    st.subheader("Confusion Matrix")
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=np.unique(true_classes), yticklabels=np.unique(true_classes))
    st.pyplot()

    # Print classification report
    st.subheader("Classification Report")
    class_report = classification_report(true_classes, predicted_classes)
    st.text(class_report)