import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from util.helpers import preprocess_data, save_model_and_history_and_details


# Load the Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Preprocess the data
train_images, train_labels, test_images, test_labels = preprocess_data(train_images, test_images, train_labels,
                                                                       test_labels)

# Build the model
model = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')  # 10 classes for the Fashion MNIST dataset
])
# Set up the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# Compile the model
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=[
                  tf.keras.metrics.Precision(name='precision'),
                  tf.keras.metrics.Recall(name='recall')
              ])

# Set up EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss',  # Can change to 'val_accuracy' if you want to monitor accuracy
    patience=5,  # Number of epochs to wait after no improvement
    restore_best_weights=True  # Restore the best weights of the model
)

# Train the model
history = model.fit(
    train_images,
    train_labels,
    epochs=100,  # Number of epochs can be greater than actual epochs due to EarlyStopping
    validation_split=0.3,  # Split data for validation
    callbacks=[early_stopping]
)
# Save the model after training
save_model_and_history_and_details(model, history, test_images, test_labels)
