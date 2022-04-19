# import cv2
import easyocr
import cv2
from pylab import rcParams


def get_string(img_path):
    # rcParams["figure.figsize"] = 10, 10
    # Recognize text with tesseract for python
    # img_path = r".\noise_free_img.jpeg"
    img = cv2.imread(img_path)
    # print(get_string(img))
    # result = pytesseract.image_to_string(img, lang="eng")
    reader = easyocr.Reader(["en"])
    output = reader.readtext(img)
    text = ""
    for i in output:
        text += i[1] + " "
    return text

