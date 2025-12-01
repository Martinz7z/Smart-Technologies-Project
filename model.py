import tensorflow as tf
from tensorflow.keras import layers, models

def create_model():
    """Create a CNN model for 24-class image classification"""
    print("Creating CNN model for 24 classes...")
    
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        
        # Output layer for 24 classes
        layers.Dense(24, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Show model summary
    print("Model created successfully!")
    model.summary()
    
    return model

def create_model_with_dropout():
    """Create a CNN model with dropout to prevent overfitting"""
    print("Creating CNN model with dropout...")
    
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Flatten and dense layers with dropout
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),  # Dropout to prevent overfitting
        
        # Output layer for 24 classes
        layers.Dense(24, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("Model with dropout created successfully!")
    model.summary()
    
    return model

# Test the model creation
if __name__ == "__main__":
    print("=== Testing Model Creation ===")
    model = create_model()
    print("\n=== Testing Model with Dropout ===")
    model_with_dropout = create_model_with_dropout()