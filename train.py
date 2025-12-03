import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from data import combine_datasets

def prepare_data_balanced():
    """Load data and make validation set balanced"""
    print("Loading dataset...")
    x_train, y_train, x_test, y_test, class_names = combine_datasets()
    
    # Make pixel values between 0 and 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Split training data into train and validation
    # Make sure validation has all classes
    x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
        x_train, y_train, 
        test_size=0.2, 
        random_state=42,
        stratify=y_train  # Keep same class mix
    )
    
    print(f"Training data: {x_train_split.shape}")
    print(f"Validation data: {x_val_split.shape}")
    print(f"Test data: {x_test.shape}")
    
    return x_train_split, y_train_split, x_val_split, y_val_split, x_test, y_test, class_names

def create_better_model():
    """Make a CNN model with dropout to prevent overfitting"""
    print("Making better model...")
    
    model = tf.keras.Sequential([
        # First block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Second block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Third block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Final layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        
        # Output for 24 classes
        tf.keras.layers.Dense(24, activation='softmax')
    ])
    
    # Set up model for training
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("Model ready!")
    model.summary()
    
    return model

def train_better_model():
    """Train the model with better settings"""
    print("\n=== Training Better Model ===")
    
    # Get data
    x_train, y_train, x_val, y_val, x_test, y_test, class_names = prepare_data_balanced()
    
    # Make model
    model = create_better_model()
    
    # Stop early if not improving
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Lower learning rate if stuck
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001
    )
    
    # Train
    print("\nTraining model...")
    history = model.fit(
        x_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Save model
    model.save('cifar24_better_model.keras')
    print("Model saved")
    
    # Test
    print("\nTesting model...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Make charts
    make_charts(history, 'training_charts.png')
    
    return model, history, test_acc

def make_charts(history, filename='training_charts.png'):
    """Make accuracy and loss charts"""
    plt.figure(figsize=(12, 4))
    
    # Accuracy chart
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Loss chart
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# Run everything
if __name__ == "__main__":
    print("=== Training 24-Class Model ===")
    print("Fixing overfitting...")
    
    model, history, test_acc = train_better_model()
    print(f"\nDone!")
    print(f"Final test accuracy: {test_acc:.4f}")
    print("\nWhat we changed:")
    print("- More dropout layers")
    print("- Balanced validation")
    print("- Stop early if not improving")