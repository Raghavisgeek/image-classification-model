# image-classification-model

*COMPANY*: CODTECH IT SOLUTIONS

 *NAME*: RAGHAV PANDEY
 
 *INTERN ID*: CT04DF122
 
 *DOMAIN*: MACHINE LEARNING
 
 *DURATION*: 4 WEEKS
 
 *MENTOR*: NEELA SANTOSH

 This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify images of cats and dogs. The dataset is sourced from Kaggle: [Dogs vs Cats Dataset](https://www.kaggle.com/datasets/salader/dogs-vs-cats). This project is part of an internship task and demonstrates the foundational application of deep learning in image classification.

## 📁 Dataset

- **Source**: [Kaggle - salader/dogs-vs-cats](https://www.kaggle.com/datasets/salader/dogs-vs-cats)
- **Images**: JPEG files of cats and dogs in a 256x256 resized format.
- **Binary classification**: `1 = Dog`, `0 = Cat`

## 🔧 Model Architecture

The CNN model was created using the Keras Sequential API and includes:

```
Input shape: (256, 256, 3)

1. Conv2D (32 filters, kernel_size=3x3) + ReLU + BatchNormalization + MaxPooling2D
2. Conv2D (64 filters, kernel_size=3x3) + ReLU + BatchNormalization + MaxPooling2D
3. Conv2D (128 filters, kernel_size=3x3) + ReLU + BatchNormalization + MaxPooling2D
4. Flatten layer
5. Dense (128) + ReLU + Dropout(0.1)
6. Dense (64) + ReLU + Dropout(0.1)
7. Output: Dense (1) + Sigmoid (for binary classification)
```

## 🧪 Training

- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy
- **Validation Split**: Typically 20% of training set
- **Data Augmentation**: Optional, but recommended for higher accuracy
- **Epochs**: Configurable (e.g., 10-20)
- **Batch Size**: Usually 32

## ⚙️ Setup Instructions

1. Install dependencies:
```bash
pip install tensorflow keras numpy matplotlib
```

2. Download the dataset via Kaggle CLI:
```bash
kaggle datasets download -d salader/dogs-vs-cats
unzip dogs-vs-cats.zip
```

3. Run the training script (if using Jupyter):
```python
model.fit(train_data, epochs=10, validation_data=val_data)
```

## ✅ Results

- **Validation Accuracy**: ~80% (varies depending on augmentation and training epochs)
- **Loss trend**: Decreasing over time (use EarlyStopping if overfitting)

## 📌 Key Learnings

This project served as an introduction to:
- Building CNNs with TensorFlow/Keras
- Image preprocessing and augmentation
- Evaluating classification models using validation accuracy
- Overfitting control with Dropout and BatchNormalization

## 🔮 Future Improvements

- Add more Conv layers and play with kernel sizes
- Use data augmentation (rotation, flip, zoom) to boost performance
- Apply learning rate schedulers or transfer learning (e.g., MobileNetV2, ResNet50)

---

Created by Raghav Pandey
