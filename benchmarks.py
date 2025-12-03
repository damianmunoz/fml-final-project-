# ============================================================================
# HANDWRITTEN TEXT RECOGNITION USING TRADITIONAL ML
# ============================================================================
# Complete pipeline for recognizing handwritten text using SVM, k-NN, Random Forest
# Copy this entire script into a single Colab cell and run!
# ============================================================================

# ----------------------------------------------------------------------------
# INSTALLATION & IMPORTS
# ----------------------------------------------------------------------------
print("Installing packages...")
import subprocess
import sys

packages = ['opencv-python-headless', 'scikit-image', 'scikit-learn', 'matplotlib', 'seaborn', 'tqdm']
for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

print("Packages installed!\n")

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import os
from pathlib import Path
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

from skimage import io, color, filters, measure
from skimage.feature import hog
from skimage.morphology import binary_closing, binary_opening
from skimage.filters import threshold_otsu

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("All libraries imported!\n")

# ----------------------------------------------------------------------------
# PREPROCESSING MODULE
# ----------------------------------------------------------------------------
class ImagePreprocessor:
    """
    A class for preprocessing handwritten text images.
    Includes binarization, denoising, and normalization.
    """
    
    @staticmethod
    def binarize_otsu(image):
        """Apply Otsu's thresholding for binarization"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        threshold = threshold_otsu(image)
        binary = image > threshold
        return binary.astype(np.uint8) * 255
    
    @staticmethod
    def denoise_gaussian(image, sigma=1.0):
        """Apply Gaussian filtering to reduce noise"""
        return filters.gaussian(image, sigma=sigma)
    
    @staticmethod
    def denoise_median(image, size=3):
        """Apply median filtering to reduce noise"""
        return cv2.medianBlur(image.astype(np.uint8), size)
    
    @staticmethod
    def normalize_image(image):
        """Normalize image to [0, 1] range"""
        return (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    @staticmethod
    def preprocess_pipeline(image, denoise_method='gaussian'):
        """Complete preprocessing pipeline"""
        image = ImagePreprocessor.normalize_image(image)
        
        if denoise_method == 'gaussian':
            image = ImagePreprocessor.denoise_gaussian(image)
        elif denoise_method == 'median':
            image = ImagePreprocessor.denoise_median((image * 255).astype(np.uint8))
            image = image / 255.0
        
        image = ImagePreprocessor.binarize_otsu((image * 255).astype(np.uint8))
        return image

print("Preprocessing module loaded!\n")

# ----------------------------------------------------------------------------
# 🎯 FEATURE EXTRACTION MODULE
# ----------------------------------------------------------------------------
class FeatureExtractor:
    """
    Extract features from handwritten character images.
    Supports HOG, zoning, and pixel density features.
    """
    
    @staticmethod
    def extract_hog_features(image, orientations=9, pixels_per_cell=(8, 8), 
                            cells_per_block=(2, 2)):
        """Extract HOG (Histogram of Oriented Gradients) features"""
        image = image.reshape(28, 28)
        features = hog(image, orientations=orientations, 
                      pixels_per_cell=pixels_per_cell,
                      cells_per_block=cells_per_block, 
                      block_norm='L2-Hys')
        return features
    
    @staticmethod
    def extract_zoning_features(image, zones=(4, 4)):
        """Extract zoning features by dividing image into grid"""
        image = image.reshape(28, 28)
        h, w = image.shape
        zone_h, zone_w = h // zones[0], w // zones[1]
        
        features = []
        for i in range(zones[0]):
            for j in range(zones[1]):
                zone = image[i*zone_h:(i+1)*zone_h, j*zone_w:(j+1)*zone_w]
                features.append(np.mean(zone))
        
        return np.array(features)
    
    @staticmethod
    def extract_pixel_density(image):
        """Extract overall pixel density and distribution features"""
        image = image.reshape(28, 28)
        features = [
            np.mean(image),
            np.std(image),
            np.sum(image > 0) / image.size,
        ]
        return np.array(features)

print("Feature extraction module loaded!\n")

# ----------------------------------------------------------------------------
# LOAD DATASET
# ----------------------------------------------------------------------------
print("Downloading MNIST dataset (this may take a minute)...")

try:
    mnist = fetch_openml('mnist_784', version=1, cache=True, parser='auto')
    X, y = mnist["data"], mnist["target"]
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Dataset loaded: {X.shape[0]} samples")
    print(f"   Image size: 28x28 pixels")
    print(f"   Classes: {len(np.unique(y))}\n")
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please check your internet connection and try again.")
    sys.exit(1)

# ----------------------------------------------------------------------------
# VISUALIZE SAMPLES
# ----------------------------------------------------------------------------
def visualize_samples(X, y, n_samples=10, title="Sample Handwritten Digits"):
    """Visualize random samples from the dataset"""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    indices = np.random.choice(len(X), n_samples, replace=False)
    
    for idx, ax in enumerate(axes.flat):
        image = X[indices[idx]].reshape(28, 28)
        ax.imshow(image, cmap='gray')
        ax.set_title(f"Label: {y[indices[idx]]}", fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

print("Visualizing sample images...")
visualize_samples(X, y)

# ----------------------------------------------------------------------------
# DEMONSTRATE PREPROCESSING
# ----------------------------------------------------------------------------
def demonstrate_preprocessing(X, idx=0):
    """Visualize preprocessing steps"""
    sample = X[idx].reshape(28, 28)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('Preprocessing Pipeline Demonstration', fontsize=14, fontweight='bold')
    
    # Original
    axes[0].imshow(sample, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Normalized
    normalized = ImagePreprocessor.normalize_image(sample)
    axes[1].imshow(normalized, cmap='gray')
    axes[1].set_title('Normalized')
    axes[1].axis('off')
    
    # Denoised
    denoised = ImagePreprocessor.denoise_gaussian(normalized)
    axes[2].imshow(denoised, cmap='gray')
    axes[2].set_title('Denoised (Gaussian)')
    axes[2].axis('off')
    
    # Binarized
    binary = ImagePreprocessor.binarize_otsu((denoised * 255).astype(np.uint8))
    axes[3].imshow(binary, cmap='gray')
    axes[3].set_title('Binarized (Otsu)')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()

print("Demonstrating preprocessing pipeline...")
demonstrate_preprocessing(X)

# ----------------------------------------------------------------------------
# EXTRACT FEATURES
# ----------------------------------------------------------------------------
print("Extracting features from samples...")

# Use subset for faster processing
n_samples = 10000
X_subset = X[:n_samples]
y_subset = y[:n_samples]

# Extract HOG features
X_features = []
for i in tqdm(range(len(X_subset)), desc="Extracting HOG features"):
    features = FeatureExtractor.extract_hog_features(X_subset[i])
    X_features.append(features)

X_features = np.array(X_features)

print(f"Feature extraction complete!")
print(f"   Original shape: {X_subset.shape}")
print(f"   Feature shape: {X_features.shape}\n")

# ----------------------------------------------------------------------------
# SPLIT DATA
# ----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_subset, test_size=0.2, random_state=42, stratify=y_subset
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f" Dataset split:")
print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}\n")

# ----------------------------------------------------------------------------
# TRAIN CLASSIFIERS
# ----------------------------------------------------------------------------
classifiers = {
    'SVM (Linear)': SVC(kernel='linear', random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', random_state=42),
    'k-NN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}

results = {}

for name, clf in classifiers.items():
    print(f"\nTraining {name}...")
    
    # Train
    clf.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = clf.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'model': clf,
        'predictions': y_pred
    }
    
    print(f"{name} trained!")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1-Score: {f1:.4f}")

# ----------------------------------------------------------------------------
# VISUALIZE RESULTS
# ----------------------------------------------------------------------------
results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'Precision': [results[m]['precision'] for m in results.keys()],
    'Recall': [results[m]['recall'] for m in results.keys()],
    'F1-Score': [results[m]['f1_score'] for m in results.keys()]
})

print("\nModel Performance Comparison:\n")
print(results_df.to_string(index=False))

# Plot comparison
fig, ax = plt.subplots(figsize=(12, 6))
results_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score']].plot(
    kind='bar', ax=ax, rot=45
)
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_ylabel('Score')
ax.set_ylim([0, 1])
ax.legend(loc='lower right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Confusion matrix
best_model_name = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
y_pred_best = results[best_model_name]['predictions']

cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

print(f"\nBest Model: {best_model_name}")
print(f"   Accuracy: {results[best_model_name]['accuracy']:.4f}")

# ----------------------------------------------------------------------------
# INTERACTIVE PREDICTIONS
# ----------------------------------------------------------------------------
def predict_sample(model, sample_idx):
    """Predict and visualize a single sample"""
    features = X_test_scaled[sample_idx].reshape(1, -1)
    prediction = model.predict(features)[0]
    true_label = y_test[sample_idx]
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    original_image = X_subset[len(X_train) + sample_idx].reshape(28, 28)
    
    ax.imshow(original_image, cmap='gray')
    ax.set_title(f'True: {true_label} | Predicted: {prediction}\n' + 
                f'{"Correct" if str(prediction) == str(true_label) else "Incorrect"}',
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return prediction, true_label

print("\nTesting random samples:\n")
for i in range(5):
    random_idx = np.random.randint(0, len(X_test))
    pred, true = predict_sample(results[best_model_name]['model'], random_idx)

# ----------------------------------------------------------------------------
# SAVE MODELS
# ----------------------------------------------------------------------------
import pickle

with open('best_model.pkl', 'wb') as f:
    pickle.dump(results[best_model_name]['model'], f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n Models saved successfully!")
print(f"   - best_model.pkl ({best_model_name})")
print("   - scaler.pkl")

print("\n" + "="*80)
print("PROJECT COMPLETE!")
print("="*80)
