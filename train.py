import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data import combine_datasets
from model import create_model_with_dropout

def prepare_data():
    """Load and prepare the dataset for training"""
    print("Loading dataset...")
    x_train, y_train, x_test, y_test, class_names = combine_datasets()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    
    return x_train, y_train, x_test, y_test, class_names

def train_model():
    """Train the CNN model on the dataset"""
    print("\n=== Starting Model Training ===")
    
    # Prepare data
    x_train, y_train, x_test, y_test, class_names = prepare_data()
    
    # Create model
    model = create_model_with_dropout()
    
    # Train the model
    print("\nTraining model...")
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Save the trained model
    model.save('cifar24_model.h5')
    print("Model saved as 'cifar24_model.h5'")
    
    # Evaluate on test data
    print("\nEvaluating on test data...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Plot training history
    plot_training_history(history)
    
    return model, history, test_acc

def plot_training_history(history):
    """Plot training and validation accuracy/loss"""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# Main execution
if __name__ == "__main__":
    print("=== CIFAR-10/100 24-Class Classification Training ===")
    model, history, test_acc = train_model()
    print(f"\nTraining completed!")
    print(f"Final test accuracy: {test_acc:.4f}")