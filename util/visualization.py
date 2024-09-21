import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Function to plot the Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, labels):
    """
    Plots the confusion matrix for the true and predicted labels.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    labels (list): List of label names.

    Returns:
    plt: The matplotlib plot object.
    """
    cm = confusion_matrix(y_true, y_pred)  # Compute confusion matrix
    plt.figure(figsize=(10, 5))  # Set figure size
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)  # Create heatmap
    plt.xlabel('Predicted')  # Label for x-axis
    plt.ylabel('True')  # Label for y-axis
    plt.title('Confusion Matrix')  # Title of the plot
    return plt  # Return the plot


# Function to plot Loss over epochs
def plot_loss(history):
    """
    Plots the training and validation loss over epochs.

    Parameters:
    history (dict): Dictionary containing loss values.

    Returns:
    plt: The matplotlib plot object.
    """
    plt.figure(figsize=(10, 5))  # Set figure size
    plt.plot(history['loss'], label='Training Loss')  # Plot training loss
    plt.plot(history['val_loss'], label='Validation Loss')  # Plot validation loss
    plt.title('Loss Over Time')  # Title of the plot
    plt.xlabel('Epochs')  # Label for x-axis
    plt.ylabel('Loss')  # Label for y-axis
    plt.legend()  # Show legend
    return plt  # Return the plot


# Function to plot Precision over epochs
def plot_precision(history):
    """
    Plots the training and validation precision over epochs.

    Parameters:
    history (dict): Dictionary containing precision values.

    Returns:
    plt: The matplotlib plot object.
    """
    plt.figure(figsize=(10, 5))  # Set figure size
    plt.plot(history['precision'], label='Training Precision')  # Plot training precision
    plt.plot(history['val_precision'], label='Validation Precision')  # Plot validation precision
    plt.title('Precision Over Time')  # Title of the plot
    plt.xlabel('Epochs')  # Label for x-axis
    plt.ylabel('Precision')  # Label for y-axis
    plt.legend()  # Show legend
    return plt  # Return the plot


# Function to plot Recall over epochs
def plot_recall(history):
    """
    Plots the training and validation recall over epochs.

    Parameters:
    history (dict): Dictionary containing recall values.

    Returns:
    plt: The matplotlib plot object.
    """
    plt.figure(figsize=(10, 5))  # Set figure size
    plt.plot(history['recall'], label='Training Recall')  # Plot training recall
    plt.plot(history['val_recall'], label='Validation Recall')  # Plot validation recall
    plt.title('Recall Over Time')  # Title of the plot
    plt.xlabel('Epochs')  # Label for x-axis
    plt.ylabel('Recall')  # Label for y-axis
    plt.legend()  # Show legend
    return plt  # Return the plot


def plot_f1(history):
    """
    Plots the training and validation F1 score over epochs using precision and recall from the training history.

    Parameters:
    history (dict): Dictionary containing precision and recall values.

    Returns:
    plt: The matplotlib plot object.
    """
    precision = history['precision']
    recall = history['recall']
    val_precision = history['val_precision']
    val_recall = history['val_recall']

    # حساب F1 score لكل فترة تدريب (Training F1 Score)
    f1_scores = [2 * (p * r) / (p + r) if (p + r) != 0 else 0 for p, r in zip(precision, recall)]

    # حساب F1 score لكل فترة تحقق (Validation F1 Score)
    val_f1_scores = [2 * (p * r) / (p + r) if (p + r) != 0 else 0 for p, r in zip(val_precision, val_recall)]

    # Plot F1 scores
    plt.figure(figsize=(10, 5))  # Set figure size
    plt.plot(f1_scores, label='Training F1 Score')  # Plot training F1 score
    plt.plot(val_f1_scores, label='Validation F1 Score')  # Plot validation F1 score
    plt.title('F1 Score Over Time')  # Title of the plot
    plt.xlabel('Epochs')  # Label for x-axis
    plt.ylabel('F1 Score')  # Label for y-axis
    plt.legend()  # Show legend
    return plt  # Return the plot
