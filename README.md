# MNIST Handwritten Digit Classification Using CNN

## Overview
This project implements a **Convolutional Neural Network (CNN)** to classify handwritten digits from the **MNIST dataset**. The model is trained using TensorFlow and achieves high accuracy with minimal training time.

## Features
- **Loads MNIST Dataset** using `tf.keras.datasets`
- **Preprocesses Data** (normalization, reshaping for CNN compatibility)
- **Builds a Simple CNN Model** (convolution, pooling, dense layers)
- **Compiles & Trains the Model** with the **Adam optimizer** and **categorical cross-entropy loss**
- **Evaluates Model Performance** on the test dataset
- **Generates Predictions** on sample images
- **Visualizes Predictions** with actual and predicted labels

## Installation
Ensure you have TensorFlow and Matplotlib installed:

```bash
pip install tensorflow matplotlib numpy
```

## Dataset
- **MNIST** contains **60,000 training** and **10,000 test** images.
- Each image is **28x28 pixels**, grayscale, and labeled (0-9).

## Model Architecture
1. **Convolutional Layer** (32 filters, 3x3 kernel, ReLU activation)
2. **MaxPooling Layer** (2x2 pooling window)
3. **Flatten Layer** (converts 2D feature maps into 1D vector)
4. **Dense Layer** (128 neurons, ReLU activation)
5. **Output Layer** (10 neurons, softmax activation for multi-class classification)

## Training
```python
model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
```
- **Epochs:** 3 (fast training with good accuracy)
- **Loss:** Sparse categorical cross-entropy
- **Optimizer:** Adam

## Evaluation
```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_acc:.2f}")
```
- **Training Accuracy:** ~92%
- **Test Accuracy:** ~91%

## Prediction & Visualization
```python
predictions = model.predict(x_test[:5])
for i in range(5):
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"True Label: {y_test[i]} | Prediction: {np.argmax(predictions[i])}")
    plt.axis('off')
    plt.show()
```
This displays 5 test images with their **true labels** and **model predictions**.

## Future Improvements
- Increase **training epochs** for better accuracy
- Add **Dropout layers** to reduce overfitting
- Implement **Data Augmentation** for robustness
- Use **Deep CNNs** (e.g., ResNet, VGG) for higher accuracy

## License
This project is open-source under the **MIT License**.

---



