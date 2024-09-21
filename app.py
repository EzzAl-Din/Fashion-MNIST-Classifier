import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from util.helpers import predict_image, load_and_preprocess_data
from util.visualization import plot_confusion_matrix, plot_loss, plot_precision, plot_recall, plot_f1
from tensorflow.keras.datasets import fashion_mnist
from sklearn.metrics import classification_report
import json

# Load the pre-trained model
model = load_model('model/saved_models/model_1/model.keras')

# Load the pre-trained model history from JSON file
with open('model/saved_models/model_1/history.json', 'r') as file:
    history_data = json.load(file)

# Load and preprocess the Fashion MNIST dataset
train_images, train_labels, test_images, test_labels = load_and_preprocess_data()

# Get predictions for test images
y_pred = model.predict(test_images)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(test_labels, axis=1)

# Define class labels
labels = ['T-shirt', 'Pants', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']


def create_layout():
    """Create the layout for the Streamlit app."""
    col1, col2 = st.columns([1, 2])

    with col1:
        display_file_uploader()

    with col2:
        display_charts()


def display_file_uploader():
    """Display the file uploader for image classification."""
    st.title("Fashion MNIST Classification")
    st.write("Upload an image for classification.")

    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=100)

        with st.spinner("Predicting..."):
            label = predict_image(model, image)

        st.write(f"Predicted label: {labels[label]}")


def display_charts():
    """Display various charts based on the user's selection."""
    option = st.selectbox("Select the metric to display:",
                          ("Confusion Matrix", "Classification Report", "Loss", "Precision", "Recall", "F1 Score"))

    if option == "Confusion Matrix":
        st.pyplot(plot_confusion_matrix(y_true_classes, y_pred_classes, labels))
    elif option == "Classification Report":
        display_classification_report(y_true_classes, y_pred_classes)
    elif option == "Loss":
        st.pyplot(plot_loss(history_data))
    elif option == "Precision":
        st.pyplot(plot_precision(history_data))
    elif option == "Recall":
        st.pyplot(plot_recall(history_data))
    elif option == "F1 Score":
        st.pyplot(plot_f1(history_data))


def display_classification_report(y_true, y_pred):
    """Display the classification report as a DataFrame."""
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True, height=492)


if __name__ == "__main__":
    create_layout()
