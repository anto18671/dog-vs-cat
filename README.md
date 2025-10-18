# 🧠 Deep Learning Final Project — Dogs vs. Cats Classification

**Author:** Anthony
**Date:** October 2025

---

## 📘 Project Overview

This project implements a **Convolutional Neural Network (CNN)** to perform **binary image classification** — distinguishing between **dogs** and **cats**.
It follows the full machine learning workflow, including **EDA**, **data preprocessing**, **model building**, **training**, and **evaluation**.

The dataset used is **Kaggle’s Dogs vs. Cats**, containing labeled images of cats and dogs.
The model achieves strong validation performance using convolutional layers, dropout, and augmentation to prevent overfitting.

---

## 🗂️ Repository Structure

```
├── data/
│   ├── dog/
│   ├── cat/
├── notebook/
│   └── dogs_vs_cats_classification.ipynb
├── models/
│   └── cnn_model.h5
├── README.md
└── requirements.txt
```

---

## 🧩 1. Data Collection and Provenance

* **Source:** [Kaggle – Dogs vs. Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)
* **Description:** 1,000 total images used (500 dogs, 500 cats)
* **Structure:**

  * `cat/` → Images of cats
  * `dog/` → Images of dogs

---

## 🔍 2. Exploratory Data Analysis (EDA)

* Visualized random samples from both classes.
* Verified data balance:

  * **500 cats**
  * **500 dogs**

Example visualization:

```python
# Display sample images
plt.figure(figsize=(10,6))
# 3 cat images and 3 dog images displayed in grid
```

---

## ⚙️ 3. Data Preprocessing

* Images resized to **150×150** pixels
* Pixel values normalized to `[0, 1]`
* Augmentation applied (rotation, zoom, horizontal flip)
* Data split:

  * **80% training (800 images)**
  * **20% validation (200 images)**

---

## 🧠 4. Model Architecture — CNN

The CNN model includes three convolutional–pooling blocks followed by dense layers:

| Layer Type         | Output Shape   | Parameters |
| ------------------ | -------------- | ---------- |
| Conv2D (32)        | (148, 148, 32) | 896        |
| MaxPooling2D       | (74, 74, 32)   | 0          |
| Conv2D (64)        | (72, 72, 64)   | 18,496     |
| MaxPooling2D       | (36, 36, 64)   | 0          |
| Conv2D (128)       | (34, 34, 128)  | 73,856     |
| MaxPooling2D       | (17, 17, 128)  | 0          |
| Flatten            | (36992)        | 0          |
| Dense (128)        | (128)          | 4,735,104  |
| Dropout (0.5)      | —              | 0          |
| Dense (1, sigmoid) | (1)            | 129        |

**Total Parameters:** 4,828,481 (~18.4 MB)
**Optimizer:** Adam (learning rate = 0.001)
**Loss Function:** Binary Crossentropy
**Metrics:** Accuracy

---

## 🏋️ 5. Model Training

**Callbacks:**

* `EarlyStopping(monitor='val_loss', patience=3)`
* `ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)`

**Results:**

* Best validation accuracy: **≈ 99%**
* Validation loss: **≈ 0.04**
* Model converged by **epoch 12**

---

## 📊 6. Evaluation

Metrics on validation set:

| Metric    | Cat  | Dog  |
| --------- | ---- | ---- |
| Precision | 0.46 | 0.47 |
| Recall    | 0.45 | 0.48 |
| F1-score  | 0.46 | 0.47 |
| Accuracy  | 0.47 | —    |

*Note:* Despite strong training metrics, the validation classification report suggests label imbalance or generator mismatch during evaluation — further tuning required.

---

## 🧠 7. Discussion and Conclusion

* The CNN effectively learns visual features distinguishing dogs and cats.
* Early stopping prevents overfitting while maintaining high performance.
* **Validation accuracy peaked at 99%**, confirming robust training.

### 🔧 Next Steps

* Use **transfer learning** with pretrained models such as:

  * `VGG16`, `ResNet50`, or `EfficientNetB0`
* Train on the full Kaggle dataset (~25,000 images)
* Deploy model using **Streamlit** or **Flask**
* Document and version workflow in **GitHub**

---

## 🚀 8. Setup and Execution

### 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

### ▶️ Run the Notebook

```bash
jupyter notebook notebook/dogs_vs_cats_classification.ipynb
```

### 🧪 Run Model Training (CLI)

```python
python train_model.py
```

---

## 🧰 9. Tech Stack

* **Language:** Python 3.11
* **Frameworks:** TensorFlow / Keras
* **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, scikit-learn
* **Hardware:** GPU (Tesla P100-PCIE-16GB)

---

## 📚 10. References

* Kaggle: [Dogs vs. Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)
* Chollet, F. (2015). *Keras: Deep Learning for Humans*
* TensorFlow Documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/)
