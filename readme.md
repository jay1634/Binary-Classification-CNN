# Binary Image Classification

A CNN-based binary image classifier built with TensorFlow/Keras.

## Features

- Binary classification using CNN
- Data augmentation
- Training visualization
- Single image prediction

## Requirements

```bash
pip install numpy pandas tensorflow matplotlib
```

## Dataset Structure

```
data/
├── class1/
│   └── images...
└── class2/
    └── images...
```

## Installation

```bash
git clone https://github.com/yourusername/binary-classifier.git
cd binary-classifier
pip install -r requirements.txt
```

## Usage

Update the data path:
```python
data_path = r"path/to/your/data"
```

Train the model:
```bash
python classifier.py
```

Predict on new image:
```python
img = load_img(r"path/to/image.jpg", target_size=(128,128))
```

## Model

- 3 convolutional blocks
- Batch normalization
- Dropout regularization
- Adam optimizer
- Binary crossentropy loss

## Output

- Visualizes training/validation accuracy and loss
- Predicts class based on probability threshold (0.5)

## Author

[GitHub Profile](https://github.com/jay1634)

## Acknowledgments

- TensorFlow/Keras documentation
- Dataset source: (https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)

---

**Note**: Update the file paths and hyperparameters according to your specific setup and requirements.