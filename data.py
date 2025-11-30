import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def load_cifar10_data():
    """Load CIFAR-10 dataset and keep only the classes we need"""
    print("Loading CIFAR-10 data...")
    
    # Load the full CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # CIFAR-10 class names in order
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Classes we want to keep (their positions in the list)
    wanted_classes = [1, 2, 3, 4, 5, 7, 9]  # automobile, bird, cat, deer, dog, horse, truck
    wanted_class_names = [class_names[i] for i in wanted_classes]
    
    print(f"Keeping these classes: {wanted_class_names}")
    
    # Filter training data
    train_mask = np.isin(y_train.flatten(), wanted_classes)
    x_train_filtered = x_train[train_mask]
    y_train_filtered = y_train[train_mask]
    
    # Filter test data
    test_mask = np.isin(y_test.flatten(), wanted_classes)
    x_test_filtered = x_test[test_mask]
    y_test_filtered = y_test[test_mask]
    
    # Map old labels to new labels (0-6)
    label_mapping = {old: new for new, old in enumerate(wanted_classes)}
    y_train_mapped = np.vectorize(label_mapping.get)(y_train_filtered)
    y_test_mapped = np.vectorize(label_mapping.get)(y_test_filtered)
    
    print(f"Filtered training data: {x_train_filtered.shape}")
    print(f"Filtered test data: {x_test_filtered.shape}")
    
    return x_train_filtered, y_train_mapped, x_test_filtered, y_test_mapped, wanted_class_names

def load_cifar100_data():
    """Load CIFAR-100 dataset and keep only the classes we need"""
    print("Loading CIFAR-100 data...")
    
    # Load both fine and coarse labels
    (x_train, y_train_fine), (x_test, y_test_fine) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
    (_, y_train_coarse), (_, y_test_coarse) = tf.keras.datasets.cifar100.load_data(label_mode='coarse')
    
    # CIFAR-100 fine class names
    fine_classes = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm'
    ]
    
    # CIFAR-100 coarse (superclass) names
    coarse_classes = [
        'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
        'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
        'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores',
        'medium_mammals', 'non-insect_invertebrates', 'people', 'reptiles', 'small_mammals', 'trees',
        'vehicles_1', 'vehicles_2'
    ]
    
    # Fine classes we want to keep
    wanted_fine_classes = [
        'cattle', 'fox', 'baby', 'boy', 'girl', 'man', 'woman',
        'rabbit', 'squirrel', 'bicycle', 'bus', 'motorcycle',
        'pickup_truck', 'train', 'lawn_mower', 'tractor'
    ]
    
    # Get indices for fine classes
    wanted_indices = [fine_classes.index(cls) for cls in wanted_fine_classes]
    
    # For trees superclass, we'll take maple_tree, oak_tree, palm_tree, pine_tree, willow_tree
    tree_classes = ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree']
    tree_indices = [fine_classes.index(cls) for cls in tree_classes]
    
    # Combine all wanted indices (fine classes + one tree class)
    all_wanted_indices = wanted_indices + [tree_indices[0]]  # Just take maple_tree for now
    
    print(f"Keeping these CIFAR-100 classes: {wanted_fine_classes + ['maple_tree (for trees)']}")
    
    # Filter data
    train_mask = np.isin(y_train_fine.flatten(), all_wanted_indices)
    test_mask = np.isin(y_test_fine.flatten(), all_wanted_indices)
    
    x_train_filtered = x_train[train_mask]
    y_train_filtered = y_train_fine[train_mask]
    x_test_filtered = x_test[test_mask]
    y_test_filtered = y_test_fine[test_mask]
    
    # Map old labels to new labels (0-16)
    label_mapping = {old: new for new, old in enumerate(all_wanted_indices)}
    y_train_mapped = np.vectorize(label_mapping.get)(y_train_filtered.flatten())
    y_test_mapped = np.vectorize(label_mapping.get)(y_test_filtered.flatten())
    
    print(f"Filtered CIFAR-100 training data: {x_train_filtered.shape}")
    print(f"Filtered CIFAR-100 test data: {x_test_filtered.shape}")
    
    return x_train_filtered, y_train_mapped, x_test_filtered, y_test_mapped, wanted_fine_classes + ['trees']

def combine_datasets():
    """Combine CIFAR-10 and CIFAR-100 datasets into one dataset with 24 classes"""
    print("Combining CIFAR-10 and CIFAR-100 datasets...")
    
    # Load both datasets
    x_train10, y_train10, x_test10, y_test10, classes10 = load_cifar10_data()
    x_train100, y_train100, x_test100, y_test100, classes100 = load_cifar100_data()
    
    # Combine the classes list
    all_classes = classes10 + classes100
    print(f"Combined classes: {all_classes}")
    print(f"Total classes: {len(all_classes)}")
    
    # Adjust CIFAR-100 labels to start from 7 (after CIFAR-10's 0-6)
    y_train100_adjusted = y_train100 + 7
    y_test100_adjusted = y_test100 + 7

    # Make sure all labels have the same dimensions (flatten to 1D)
    y_train10_flat = y_train10.flatten()
    y_test10_flat = y_test10.flatten()
    
    # Combine the data
    x_train_combined = np.concatenate([x_train10, x_train100])
    y_train_combined = np.concatenate([y_train10_flat, y_train100_adjusted])
    x_test_combined = np.concatenate([x_test10, x_test100])
    y_test_combined = np.concatenate([y_test10_flat, y_test100_adjusted])
    
    print(f"Combined training data: {x_train_combined.shape}")
    print(f"Combined test data: {x_test_combined.shape}")
    print(f"Training labels range: {y_train_combined.min()} to {y_train_combined.max()}")
    print(f"Test labels range: {y_test_combined.min()} to {y_test_combined.max()}")
    
    return x_train_combined, y_train_combined, x_test_combined, y_test_combined, all_classes

def explore_data(x_train, y_train, x_test, y_test, class_names):
    """Explore and visualize the dataset"""
    print("\n=== Exploring Dataset ===")
    
    # Basic statistics
    print(f"Training set: {x_train.shape[0]} images")
    print(f"Test set: {x_test.shape[0]} images")
    print(f"Image size: {x_train.shape[1:]}")
    print(f"Number of classes: {len(class_names)}")
    
    # Class distribution
    train_class_counts = np.bincount(y_train)
    test_class_counts = np.bincount(y_test)
    
    print("\nClass distribution in training set:")
    for i, count in enumerate(train_class_counts):
        print(f"  {class_names[i]}: {count} images")
    
    print("\nClass distribution in test set:")
    for i, count in enumerate(test_class_counts):
        print(f"  {class_names[i]}: {count} images")
    
    # Plot class distribution
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(len(class_names)), train_class_counts)
    plt.title('Training Set Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(class_names)), test_class_counts)
    plt.title('Test Set Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.show()
    
    # Show sample images from each class
    print("\nShowing sample images from each class...")
    plt.figure(figsize=(15, 10))
    
    for class_idx in range(len(class_names)):
        # Find first image of this class
        sample_idx = np.where(y_train == class_idx)[0][0]
        
        plt.subplot(4, 6, class_idx + 1)
        plt.imshow(x_train[sample_idx])
        plt.title(f'{class_names[class_idx]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_images.png')
    plt.show()
    
    return train_class_counts, test_class_counts

# Test section at the bottom
if __name__ == "__main__":
    print("=== Testing Combined Dataset ===")
    x_train, y_train, x_test, y_test, all_classes = combine_datasets()
    print(f"All classes: {all_classes}")
    print(f"Final dataset sizes:")
    print(f"Training: {x_train.shape} images")
    print(f"Testing: {x_test.shape} images")
    print(f"Number of classes: {len(all_classes)}")
    
    # Explore the data
    explore_data(x_train, y_train, x_test, y_test, all_classes)