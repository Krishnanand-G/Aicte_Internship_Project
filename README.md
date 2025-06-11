# Image Classification with CNN – AICTE Internship Project

Hey! 👋  
This project was done as a part of my **AICTE internship**. The goal was to build a simple image classification model using **TensorFlow** and **Keras** that could recognize different categories of grayscale images.

I used Google Colab to train a Convolutional Neural Network (CNN) on a dataset I stored in Google Drive. The pipeline includes image preprocessing, model training, evaluation, and prediction of new images.

---

## 🔧 Tech Stack

- Python 🐍  
- TensorFlow / Keras 🧠  
- Google Colab 💻  
- ImageDataGenerator for data augmentation  
- Grayscale image classification (7 classes)

---

## 📁 Dataset Handling

1. **Mounting Google Drive**  
   The dataset was a zip file stored in my Drive which was copied and extracted directly into the Colab environment.

2. **Image Preprocessing**  
   I used `ImageDataGenerator` for:
   - Rescaling pixel values
   - Random rotations
   - Zooms
   - Width/height shifts
   - Shear transformations  
   
   The images were resized to **28x28 grayscale** format.

---

## 🧠 Model Architecture

```python
model = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    BatchNormalization(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dense(7, activation='softmax')
])
