# import cv2
import os

# import numpy as np
import temp3

# import pytesseract

import easyocr

# from pylab import rcParams
# from IPython.display import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# def getTextFromImage():
#     rcParams["figure.figsize"] = 10, 10

#     reader = easyocr.Reader(["en"])
#     output = reader.readtext(cv2.imread(r"./image.jpg"))

#     text = ""
#     for i in output:
#         text += i[1] + " "
#     return text


# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def get_string(img_path):
    # Read image using opencv
    # img = cv2.imread(img_path)
    img = temp3.get_img(img_path)
    # print(img)
    # Extract the file name without the file extension
    file_name = os.path.basename(img_path).split(".")[0]
    file_name = file_name.split()[0]

    # Create a directory for outputs
    output_path = os.path.join("/", file_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    img = temp3.rescale(img)
    img = temp3.noise_removal(img)
    img = temp3.binarize(img)
    # file_name, output_path = get_string(img_path)
    # Save the filtered image in the output directory
    save_path = r"E:\VIT\GradeAid\ocr\enhanced_img.jpeg"
    # save_path = os.path.join("/", file_name + "_enhanced.jpg")
    # cv2.imwrite(save_path, img)
    temp3.saveimg(save_path, img)
    print(save_path)
    # Recognize text with tesseract for python
    # result = pytesseract.image_to_string(img, lang="eng")
    reader = easyocr.Reader(["en"])
    output = reader.readtext(temp3.get_img(r"./image.jpg"))

    text = ""
    for i in output:
        text += i[1] + " "
    return text
    # return result


if __name__ == "__main__":
    img_path = r"E:\VIT\GradeAid\ocr\sample_text_handwritten.jpeg"
    print(get_string(img_path))
