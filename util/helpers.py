from tensorflow.keras.utils import to_categorical
import numpy as np
import json
import os
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.datasets import fashion_mnist


def load_and_preprocess_data():
    """Load and preprocess the Fashion MNIST dataset.

    Returns:
        tuple: Processed training and test images and labels.
    """
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    return preprocess_data(train_images, test_images, train_labels, test_labels)


def preprocess_data(train_images, test_images, train_labels, test_labels):
    """Normalize image pixel values and convert labels to one-hot encoding.

    Parameters:
        train_images (np.ndarray): Training images.
        test_images (np.ndarray): Test images.
        train_labels (np.ndarray): Training labels.
        test_labels (np.ndarray): Test labels.

    Returns:
        tuple: Processed training and test images and labels.
    """
    # Normalize pixel values to be between 0 and 1
    train_images_processed = train_images / 255.0
    test_images_processed = test_images / 255.0

    # Convert labels to One-Hot Encoding
    train_labels_processed = to_categorical(train_labels)
    test_labels_processed = to_categorical(test_labels)
    return train_images_processed, train_labels_processed, test_images_processed, test_labels_processed


def save_model_reports(x_test, y_test, model, folder_path):
    """Save classification report and confusion matrix to a text file.

    Parameters:
        x_test (numpy.ndarray): Test data.
        y_test (numpy.ndarray): Test labels.
        model (keras.Model): The trained Keras model.
        folder_path (str): Path to the folder where the text file will be saved.
    """
    # Generate predictions for the test data
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Create classification report and confusion matrix
    class_report = classification_report(y_true_classes, y_pred_classes)
    cm = confusion_matrix(y_true_classes, y_pred_classes)

    # Save the reports in a text file
    report_file_path = os.path.join(folder_path, "model_reports.txt")
    with open(report_file_path, 'w', encoding='utf-8') as f:
        f.write("Classification Report:\n")
        f.write(class_report + '\n')
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + '\n')


def save_model_and_history_and_details(model, history, x_test, y_test):
    """Save the model and training history, along with evaluation reports.

    Parameters:
        model (keras.Model): The trained Keras model to save.
        history (keras.callbacks.History): The history object containing training metrics.
        x_test (numpy.ndarray): Test data for evaluation.
        y_test (numpy.ndarray): Test labels for evaluation.
    """
    # Construct the path to the saved_models folder
    base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "saved_models")
    os.makedirs(base_path, exist_ok=True)  # Create directories if they don't exist

    # Increment folder name to avoid overwriting
    increment = 1
    while True:
        folder_name = f"model_{increment}"
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)  # Create the folder for the model and history
            break
        increment += 1

    # Save the model and history in the folder
    model.save(os.path.join(folder_path, "model.keras"))
    with open(os.path.join(folder_path, "history.json"), 'w') as f:
        json.dump(history.history, f)

    # Save the evaluation reports
    save_model_reports(x_test, y_test, model, folder_path)


def predict_image(model, image):
    """Preprocess the uploaded image and predict its class using the trained model.

    Parameters:
        model (keras.Model): The trained Keras model for prediction.
        image (PIL.Image): The input image to be classified.

    Returns:
        int: The index of the predicted class.
    """
    processed_image = preprocess_image(image)  # Preprocess the image
    prediction = model.predict(processed_image)  # Get model predictions
    return np.argmax(prediction)  # Return the predicted class index


def preprocess_image(image_path):
    """Preprocess the image to match the model's input requirements.

    Parameters:
        image_path (PIL.Image): The image to preprocess.

    Returns:
        np.ndarray: The preprocessed image array.
    """
    # Resize and convert the image to grayscale
    image = image_path.resize((28, 28)).convert('L')
    # Convert the image to a NumPy array and normalize it
    image_array = np.array(image) / 255.0
    # Add a batch dimension
    return np.reshape(image_array, (1, 28, 28))
