# Cat vs Dog Binary Classification 🐱🐶

A deep learning project that uses Convolutional Neural Networks (CNN) to classify images as either cats or dogs. This binary classification model is built using TensorFlow/Keras and achieves robust performance through data augmentation and regularization techniques.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Dataset Structure](#dataset-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Prediction](#prediction)

## 🎯 Overview

This project implements a binary image classifier that distinguishes between cats and dogs using a custom CNN architecture. The model employs various techniques to prevent overfitting and improve generalization:
- Data augmentation
- Batch normalization
- L2 regularization
- Dropout layers
- Learning rate scheduling
- Early stopping

## ✨ Features

- **Binary Classification**: Classifies images into two categories (Cat or Dog)
- **Data Augmentation**: Implements rotation, shifting, zooming, and flipping to increase dataset diversity
- **Regularization**: Uses L2 regularization and dropout to prevent overfitting
- **Callbacks**: Learning rate reduction and early stopping for optimal training
- **Visualization**: Training/validation accuracy and loss plots
- **Single Image Prediction**: Predict on individual images

## 🛠️ Requirements

```
numpy
pandas
tensorflow>=2.0
keras
matplotlib
```

Install dependencies:
```bash
pip install numpy pandas tensorflow matplotlib
```

## 📁 Dataset Structure

Organize your dataset in the following structure:

```
data/
├── cat/
│   ├── cat1.jpg
│   ├── cat2.jpg
│   └── ...
└── dog/
    ├── dog1.jpg
    ├── dog2.jpg
    └── ...
```

The code expects images to be organized in subdirectories where each subdirectory name represents the class label.

## 💻 Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/cat-dog-classifier.git
cd cat-dog-classifier
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Update the `data_path` variable in the script to point to your dataset location:
```python
data_path = r"path/to/your/data"
```

## 🚀 Usage

### Training the Model

Run the script to train the model:
```bash
python cat_dog_classifier.py
```

The model will:
- Load and preprocess the data
- Apply data augmentation to training set
- Train for up to 25 epochs (with early stopping)
- Save the best model weights
- Display training/validation metrics

### Making Predictions

To predict on a single image, update the image path:
```python
img = load_img(r"path/to/your/image.jpg", target_size=(128,128))
```

Output will be either "CAT" or "DOG" based on the prediction.

## 🏗️ Model Architecture

The CNN architecture consists of:

| Layer | Parameters |
|-------|-----------|
| Conv2D (32 filters) | 3x3 kernel, ReLU, L2 regularization |
| Batch Normalization | - |
| MaxPooling2D | 2x2 |
| Conv2D (64 filters) | 3x3 kernel, ReLU, L2 regularization |
| Batch Normalization | - |
| MaxPooling2D | 2x2 |
| Conv2D (128 filters) | 3x3 kernel, ReLU, L2 regularization |
| Batch Normalization | - |
| Dropout | 40% |
| MaxPooling2D | 2x2 |
| Flatten | - |
| Dense | 128 units, ReLU |
| Dropout | 50% |
| Dense (Output) | 1 unit, Sigmoid |

**Total Parameters**: ~1.5M trainable parameters

## 🎓 Training

### Hyperparameters
- **Image Size**: 128x128 pixels
- **Batch Size**: 32
- **Learning Rate**: 1e-4 (Adam optimizer)
- **Max Epochs**: 25
- **Validation Split**: 20%

### Data Augmentation
Training images undergo random transformations:
- Rotation: ±25°
- Width/Height shift: 20%
- Shear: 15%
- Zoom: 25%
- Horizontal flip
- Brightness: 0.7-1.3x

### Callbacks
- **ReduceLROnPlateau**: Reduces learning rate by 0.3x when validation loss plateaus
- **EarlyStopping**: Stops training if validation loss doesn't improve for 4 epochs

## 📊 Results

After training, the script generates two plots:
1. **Training vs Validation Accuracy**
2. **Training vs Validation Loss**

These visualizations help identify overfitting and model performance.

## 🔮 Prediction

The model outputs a probability score:
- **> 0.5**: Classified as Dog
- **≤ 0.5**: Classified as Cat

Example:
```python
prediction = model.predict(img)[0][0]
if prediction > 0.5:
    print("DOG")
else:
    print("CAT")
```

## 📝 Notes

- The model uses binary crossentropy loss for binary classification
- Input images are automatically resized to 128x128
- All images are normalized to [0, 1] range
- The validation set uses no augmentation for unbiased evaluation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

[GitHub Profile](https://github.com/jay1634)

## 🙏 Acknowledgments

- TensorFlow/Keras documentation
- Dataset source: (https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)

---

**Note**: Update the file paths and hyperparameters according to your specific setup and requirements.