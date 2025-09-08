import tkinter as tk
import PIL
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import os

# Check if model file exists before importing TensorFlow
model_path = "mnist_model.h5"
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found. Please check the path.")
    model = None
else:
    # Import TensorFlow only if model exists
    from tensorflow.keras.models import load_model
    try:
        model = load_model(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Handwritten Digit Recognizer")
        self.geometry("400x400")
        self.canvas = tk.Canvas(self, width=280, height=280, bg="white")
        self.canvas.pack(pady=10)

        self.button_frame = tk.Frame(self)
        self.button_frame.pack()

        self.predict_btn = tk.Button(self.button_frame, text="Predict", command=self.predict)
        self.predict_btn.grid(row=0, column=0, padx=10)

        self.clear_btn = tk.Button(self.button_frame, text="Clear", command=self.clear)
        self.clear_btn.grid(row=0, column=1, padx=10)

        self.result_label = tk.Label(self, text="", font=("Arial", 16))
        self.result_label.pack(pady=10)

        # Create a white image to draw on
        self.image = PIL.Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)

        # FIX: bind valid mouse events
        self.canvas.bind("<B1-Motion>", self.paint)   # Drag with left mouse button
        self.canvas.bind("<Button-1>", self.paint)    # Click with left mouse button

    def paint(self, event):
        x, y = event.x, event.y
        # Draw on Tkinter canvas
        self.canvas.create_oval(x-8, y-8, x+8, y+8, fill="black", outline="black")
        # Draw on PIL image
        self.draw.ellipse((x-8, y-8, x+8, y+8), fill=0)

    def clear(self):
        self.canvas.delete("all")
        # Reset PIL image
        self.image = PIL.Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="")

    def predict(self):
        if model is None:
            self.result_label.config(text="Model not loaded")
            return

        try:
            # Preprocess the image
            img = self.image.resize((28, 28))
            img = ImageOps.invert(img)  # Invert: background=0, digit=1
            img = np.array(img) / 255.0
            img = img.reshape(1, 28, 28, 1)  # Add batch + channel dim

            # Run prediction
            prediction = model.predict(img, verbose=0)
            digit = np.argmax(prediction)
            confidence = np.max(prediction)

            self.result_label.config(text=f"Predicted Digit: {digit} ({confidence:.2f})")
        except Exception as e:
            self.result_label.config(text=f"Error: {str(e)}")


# Run App
if __name__ == "__main__":
    app = App()
    app.mainloop()
