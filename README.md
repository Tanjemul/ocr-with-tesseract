# ocr-with-tesseract
The intention of this project is to show how to work with OCR with [Tesseract](https://github.com/tesseract-ocr/tesseract/blob/main/README.md) Library using python Flask. 

### Features

- Image uploading: Upload any image
- Process image to improve OCR quality.
- Extract text 

### Tech Stack
- Python 3.12.8
- Flask 3.1.0   
- Werkzeug 3.1.3
- pytesseract
- pillow
- numpy (optional, for further image processing)
- opencv-python (optional, for further image processing)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Tanjemul/ocr-with-tesseract.git

2. Navigate to the project directory: cd \ocr-with-tesseract-main

3. Install dependencies:
    ```bash
    pip install flask
    pip install pytesseract pillow
    pip install pytesseract
    pip install numpy opencv-python
    pip install numpy

4. Run the project
    ```bash
    set FLASK_APP=app
    flask run
On local machine, open browser and go to: http://127.0.0.1:5000

     

