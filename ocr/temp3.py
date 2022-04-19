import cv2
import os
import numpy as np

# import pytesseract

# import easyocr

# from pylab import rcParams
# from IPython.display import Image

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# def getTextFromImage():
#     rcParams["figure.figsize"] = 10, 10

#     reader = easyocr.Reader(["en"])
#     output = reader.readtext(cv2.imread(r"./image.jpg"))

#     text = ""
#     for i in output:
#         text += i[1] + " "
#     return text


# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def get_img(img_path):
    img = cv2.imread(img_path)
    return img


def saveimg(save_path, img):
    cv2.imwrite(save_path, img)


def get_string(img_path):
    # Read image using opencv
    img = cv2.imread(img_path)
    # print(img)
    # Extract the file name without the file extension
    file_name = os.path.basename(img_path).split(".")[0]
    file_name = file_name.split()[0]

    # Create a directory for outputs
    output_path = os.path.join("/", file_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    img = rescale(img)
    img = noise_removal(img)
    img = binarize(img)
    # file_name, output_path = get_string(img_path)
    # Save the filtered image in the output directory
    save_path = r"E:\VIT\GradeAid\ocr\enhanced_img.jpeg"
    # save_path = os.path.join("/", file_name + "_enhanced.jpg")
    cv2.imwrite(save_path, img)
    print(save_path)
    # Recognize text with tesseract for python
    # result = pytesseract.image_to_string(img, lang="eng")
    # reader = easyocr.Reader(["en"])
    # output = reader.readtext(cv2.imread(r"./image.jpg"))

    # text = ""
    # for i in output:
    #     text += i[1] + " "
    # return text
    # return result


def rescale(img):
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    return img


def noise_removal(img):
    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    # Apply blur to smooth out the edges
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img


def binarize(img):
    # Apply threshold to get image with only b&w (binarization)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return img


if __name__ == "__main__":
    img_path = r"E:\VIT\GradeAid\ocr\sample_text_handwritten.jpeg"
    print(get_string(img_path))
