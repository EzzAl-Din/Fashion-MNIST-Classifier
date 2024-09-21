# Fashion MNIST Classifier

## Overview

This project is a machine learning application that classifies images from the Fashion MNIST dataset using a trained Keras model. It provides an interactive interface for users to upload images and view the model's predictions, along with various evaluation metrics and visualizations. The application is built with Streamlit and showcases the effectiveness of deep learning in image classification tasks.

## Features

- **Image Classification**: Upload images of clothing items for classification.
- **Model Evaluation**: Display metrics such as confusion matrix, loss, precision, recall, and F1 score.
- **Visualization**: View various charts that depict the model's performance during training and evaluation.

## Model Architecture

The model consists of:
- An input layer for 28x28 grayscale images.
- A flattening layer to convert the 2D images into 1D arrays.
- A dense layer with 256 neurons using ReLU activation.
- An output layer with 10 neurons (one for each clothing category) using softmax activation.

## Files and Functions

- **`app.py`**: Main file that runs the Streamlit application.
- **`train_model.py`**: Contains code for training the model. A total of 8 models were trained, and the 5 best performing models were selected for further evaluation and use.
- **`util/helpers.py`**: Helper functions for data loading, preprocessing, and prediction.
- **`util/visualization.py`**: Functions for visualizing evaluation metrics and performance plots.

## Getting Started

You can easily run the application directly from the deployed link or set it up locally on your machine.

### Option 1: Run the Application Online
Access the deployed application directly via the following link:  
**[Fashion MNIST Classifier - Live Demo](https://fashion-mnist-classifier.streamlit.app/)**.

### Option 2: Run the Application Locally

1. **Clone the repository**:
    ```bash
    git clone https://github.com/EzzAl-Din/Fashion-MNIST-Classifier.git
    ```

2. **Navigate to the project directory**:
    ```bash
    cd Fashion-MNIST-Classifier
    ```

3. **Install the required packages**:
    It's recommended to use a virtual environment. You can create one and install the dependencies with:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

4. **Run the main application**:
    You can run the app locally using:
    ```bash
    streamlit run app.py
    ```

## Usage

The application will prompt you to upload an image for classification. Ensure that the image is in a format compatible with the model (e.g., JPEG or PNG).

### Image Requirements:
- **Model Performance**: The model performs best on images that closely resemble the training data.
- **Background**: The training images have a black background, and the clothing items are typically white or have solid colors.
- **Optimal Conditions**: For better results, use images with clean backgrounds and clear colors. Images that deviate significantly from these conditions may result in less accurate predictions. It is recommended to use test images provided in the dataset rather than external images.

After uploading, the model will predict the label, and you can view various evaluation metrics. You can select metrics from the dropdown to visualize the model's performance, including:
- **Confusion Matrix**: Displays the counts of true vs. predicted labels, helping you understand where the model makes mistakes.
- **Loss and Accuracy**: Graphs showing how the model's loss and accuracy evolved during training.
- **Precision, Recall, and F1 Score**: Metrics that provide insights into the model's performance regarding the classification task.

### Download Test Images
To download test images for classification, use the following link:  
**[Download Fashion MNIST Test Images](#)**.

## Requirements

- Python 3.x
- Streamlit
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

You can install the required packages using the `requirements.txt` file provided.

## Contributing

Feel free to open issues or submit pull requests to improve the project.

## Contact

For any questions or feedback, please contact [ezzaldinaref@gmail.com](mailto:ezzaldinaref@gmail.com).
