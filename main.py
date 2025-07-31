import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image, ImageTk

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Load pre-trained MobileNetV2
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Preprocess image for MobileNet
def preprocess_image(frame):
    img = cv2.resize(frame, (224, 224))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return np.expand_dims(img, axis=0)

# Predict top label
def predict(frame):
    image = preprocess_image(frame)
    preds = model.predict(image)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0][0][1]
    return decoded

# GUI
class ObjectDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detector App")
        self.video_running = False

        # Buttons
        tk.Button(root, text="📸 Take Photo", command=self.open_camera).pack(pady=10)
        tk.Button(root, text="🖼️ Choose Image", command=self.choose_image).pack(pady=10)
        tk.Button(root, text="🎥 Record Video", command=self.record_video).pack(pady=10)
        tk.Button(root, text="📁 Choose Video", command=self.choose_video).pack(pady=10)

        self.label = tk.Label(root)
        self.label.pack()

    def show_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.label.configure(image=img_tk)
        self.label.image = img_tk

    def open_camera(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            self.show_image(frame)
            label = predict(frame)
            messagebox.showinfo("Prediction", f"Object: {label}")

    def choose_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if path:
            frame = cv2.imread(path)
            self.show_image(frame)
            label = predict(frame)
            messagebox.showinfo("Prediction", f"Object: {label}")

    def record_video(self):
        cap = cv2.VideoCapture(0)
        self.video_running = True
        prev_label = ""

        while self.video_running:
            ret, frame = cap.read()
            if not ret:
                break
            label = predict(frame)
            if label != prev_label:
                prev_label = label
                print("Object:", label)
            frame = cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 255, 0), 2)
            cv2.imshow("Live Object Detection - Press Q to Quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.video_running = False

    def choose_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if path:
            cap = cv2.VideoCapture(path)
            prev_label = ""
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                label = predict(frame)
                if label != prev_label:
                    prev_label = label
                    print("Object:", label)
                frame = cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 255, 0), 2)
                cv2.imshow("Video Object Detection - Press Q to Quit", frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()

# Run app
root = tk.Tk()
app = ObjectDetectorApp(root)
root.mainloop()
