import tkinter as tk
from tkinter import filedialog
import os
import cv2
import numpy as np
from imutils.perspective import four_point_transform
def scan(path):
    img = cv2.imread(path)
    WIDTH, HEIGHT = img.shape[0], img.shape[1]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    our_document_contour = np.array([[0,0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 1000:
            p = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.015*p, closed=True)
            if area > max_area and len(approx) == 4:
                our_document_contour = approx
                max_area = area
    warped = four_point_transform(img, our_document_contour.reshape(4, 2))
    return warped

def process_file():
    # Get the path of the uploaded file
    filepath = filedialog.askopenfilename(title="Select an image file", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if filepath:
        # Process the image using the scan function
        processed_image = scan(filepath)

        # Save the processed image to a new file
        filename = os.path.basename(filepath)
        result_filepath = filedialog.asksaveasfilename(initialfile=f"processed_{filename}", defaultextension=".jpg")
        if result_filepath:
            cv2.imwrite(result_filepath, processed_image)
            status_label.config(text="Image processed and saved successfully!", fg="green")

window = tk.Tk()
window.title("File Scanner")
window.geometry("400x300")
window.resizable(False, False)


window.configure(bg="white")
window.option_add("*Font", "Arial 12")


title_label = tk.Label(window, text="Image Processing Application", font=("Arial", 16, "bold"), fg="darkblue",
                       bg="white")
title_label.pack(pady=90)


upload_button = tk.Button(window, text="Upload Image", command=process_file, bg="lightblue", fg="white", width=15)
upload_button.pack(pady=(30))


status_label = tk.Label(window, text="", fg="red", bg="white")
status_label.pack(pady=10)

window.mainloop()