# Tesseract-OCR

# setting 해야 하는 라이브러리 :
# pillow
# pytesseract
# tesseract-ocr

from PIL import Image
from pytesseract import *

pytesseract.tesseract_cmd = r'C:\Users\USER\AppData\Local\Tesseract-OCR\tesseract.exe'

# im = Image.open('../img/textimg/picture05_result.PNG')

def ocrtext(im):
    text = image_to_string(im, lang='kor')
    return text

