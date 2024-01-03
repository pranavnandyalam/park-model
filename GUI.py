import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from skimage.io import imread
from skimage.transform import resize
from joblib import load
import tkinter.font as tkFont

def load_categories(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]

def predict_image(image_path, model, categories):
    img = imread(image_path)
    img_resized = resize(img, (150, 150, 3))
    img_processed = img_resized.flatten().reshape(1, -1)
    prediction = model.predict(img_processed)
    return categories[prediction[0]]

def load_image():
    global img_path, img_display
    img_path = filedialog.askopenfilename()
    if img_path:
        img = Image.open(img_path)
        img.thumbnail((250, 250))
        img_display = ImageTk.PhotoImage(img)
        image_label.config(image=img_display)
        image_label.image = img_display
        image_label_frame.config(borderwidth=2, relief="solid")

def calculate_probability(image_path, model, Categories):
    img = imread(image_path)
    img_resize = resize(img, (150, 150, 3))
    l = [img_resize.flatten()]
    probability = model.predict_proba(l)

    result = []
    for ind, val in enumerate(Categories):
        result.append((val, probability[0][ind] * 100))

    return result

def make_prediction():
    if img_path:
        result = predict_image(img_path, model, Categories)
        prob = calculate_probability(img_path, model, Categories)
        if result == 'NON_PD':
            result = "Non-Parkinson's"
        if result == 'PD':
            result = "Parkinson's"

        result_text = f'Prediction: {result}\n\n'
        for val, p in prob:
            if val == 'NON_PD':
                val = "Non-Parkinson's"
            if val == 'PD':
                val = "Parkinson's"
            result_text += f'{val}: {p:.2f}%\n'
        result_label.config(text=result_text)
        result_label_frame.config(borderwidth=3, relief="solid")

x = input("Which model would you like to use? ")
if x.lower() == "svm":
    model = load('svm_model.joblib')
if x.lower() == "lr":
    model = load('logistic_regression_model.joblib')
if x.lower() == "dtree":
    model = load('decision_tree_model.joblib')

Categories = load_categories('categories.txt')

gui = tk.Tk()
gui.title("Parkinson's Classifier")

gui.geometry("800x600")

font = tkFont.Font(family="Comfortaa", weight="bold", size=16)
load_button = tk.Button(gui, text="Load Image", command=load_image, bg="grey", fg="blue", font=font)
load_button.pack(pady=20)

image_label_frame = tk.Frame(gui, bg="white")
image_label_frame.pack(pady=20)

image_label = tk.Label(image_label_frame)
image_label.pack()

predict_button = tk.Button(gui, text="Predict", command=make_prediction, bg="grey", fg="blue", font=font)
predict_button.pack(pady=20)

result_label_frame = tk.Frame(gui, bg="grey")
result_label_frame.pack(pady=20)

result_label = tk.Label(result_label_frame, text="Prediction: None", bg="grey", fg="blue", font=font)
result_label.pack()

gui.mainloop()
