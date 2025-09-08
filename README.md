
#  Handwritten Digit Recognizer (MNIST)

This is a simple **Python GUI app** that lets you draw digits (0–9) on a canvas, and the program will recognize the digit using a trained **Convolutional Neural Network (CNN)** model on the **MNIST dataset**.

---

## Features

* Draw digits on a Tkinter canvas
* Predict the digit using a trained deep learning model (`mnist_model.h5`)
* Clear the canvas and draw again
* Shows prediction result with confidence

---

## Project Structure

```
project-folder/
│
├── num_recognizer.py     # Main GUI application
├── train_mnist.py        # Script to train and save the model
├── mnist_model.h5        # Trained model (created after running train_mnist.py)
└── README.md             # Project documentation
```

---

## Requirements

Make sure you have Python 3.9+ installed. Then install dependencies:

```bash
pip install tensorflow pillow numpy
```

Optional (for development / faster training):

```bash
pip install jupyter matplotlib
```

---

## Usage

### 1. Train the Model (first time only)

Run this to train the CNN on the MNIST dataset and save it as `mnist_model.h5`:

```bash
python train_mnist.py
```

This will download the MNIST dataset (\~11 MB), train for \~1–2 minutes on CPU, and save the model.

---

### 2. Run the GUI App

Start the handwritten digit recognizer:

```bash
python num_recognizer.py
```

* Draw a digit with your mouse.
* Click **Predict** to see the recognized digit.
* Click **Clear** to reset the canvas.

---

## How it Works

1. **Model Training (`train_mnist.py`)**

   * Uses **TensorFlow/Keras** with a CNN:

     * Conv2D → MaxPooling → Conv2D → MaxPooling → Dense → Softmax
   * Trained on MNIST dataset (60,000 training images, 10,000 test images).
   * Saves the trained model as `mnist_model.h5`.

2. **Digit Recognition (`num_recognizer.py`)**

   * Provides a Tkinter canvas for drawing digits.
   * Converts the drawing to a 28×28 grayscale image.
   * Normalizes & reshapes it to match the CNN input.
   * Model predicts the digit (0–9).

---

## Future Improvements

* Add probability distribution graph for predictions
* Improve drawing smoothing
* Allow saving and loading user-drawn digits
* Try alternative ML models (SVM, Random Forest, etc.)

---

## License

This project is open-source and free to use for learning purposes.

---
