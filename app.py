from flask import Flask, request, render_template, redirect, url_for
from PIL import Image, ImageEnhance, ImageFilter
from werkzeug.utils import secure_filename
from PIL import Image
import pytesseract
import os
import cv2
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Check if a file is uploaded
    if 'image' not in request.files:
        return render_template('index.html', extracted_text=None, uploaded_image=None, error="No file uploaded")

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', extracted_text=None, uploaded_image=None, error="No selected file")

    # Save the uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # OCR processing for Bangla
    try:
        img = Image.open(filepath)
        # Preprocess the image and save it
        processed_filepath = preprocess_image(img, app.config['UPLOAD_FOLDER'], f"processed_{filename}")
        os.remove(filepath)
        text = pytesseract.image_to_string(Image.open(processed_filepath), lang='eng+ben')
        return render_template('index.html', extracted_text=text, uploaded_image=processed_filepath)
    except Exception as e:
        return render_template('index.html', extracted_text=None, uploaded_image=None, error=f"Error during OCR: {e}")


def preprocess_image(img, output_dir, filename):
    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Convert to grayscale
        img = img.convert('L')

        # Resize for better clarity (scale up)
        img = img.resize((img.width * 2, img.height * 2), Image.Resampling.LANCZOS)

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = (ImageEnhance.Brightness(img)).enhance(5)
        img = enhancer.enhance(5)

        # Apply sharpening filter
        img = img.filter(ImageFilter.SHARPEN)
        #img = img.filter(ImageFilter.SMOOTH)
        #img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
        

        # Construct the file path
        processed_filepath = os.path.join(output_dir, filename)

        # Save the processed image
        img.save(processed_filepath)

        # Return the file path of the processed image
        return processed_filepath
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {e}")


def preprocess_image_v2(img, output_dir, filename):
    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Convert PIL image to OpenCV format (numpy array)
        img_cv = np.array(img)

        # Convert to grayscale if not already
        if len(img_cv.shape) == 3:  # If RGB or similar
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # Resize for better clarity (scale up)
        height, width = img_cv.shape
        img_cv = cv2.resize(img_cv, (width * 2, height * 2), interpolation=cv2.INTER_LANCZOS4)

        # Denoise the image using GaussianBlur
        img_cv = cv2.GaussianBlur(img_cv, (5, 5), 0)

        # Apply adaptive thresholding for binarization
        img_cv = cv2.adaptiveThreshold(
            img_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Enhance edges using morphological transformations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img_cv = cv2.morphologyEx(img_cv, cv2.MORPH_CLOSE, kernel)

        # Convert back to PIL Image
        processed_img = Image.fromarray(img_cv)

        # Construct the file path
        processed_filepath = os.path.join(output_dir, filename)

        # Save the processed image
        processed_img.save(processed_filepath)

        # Return the file path of the processed image
        return processed_filepath
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {e}")


if __name__ == '__main__':
    app.run(debug=True)
