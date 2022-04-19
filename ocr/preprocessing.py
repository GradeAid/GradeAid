import cv2

# import os
import numpy as np

# import pre_processing_1
# import get_text_from_image

# import pytesseract

# import easyocr

# from pylab import rcParams
# from IPython.display import Image

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def image_smoothening(img):
    ret1, th1 = cv2.threshold(img, 88, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3


def remove_noise_and_smooth(img):
    filtered = cv2.adaptiveThreshold(
        img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 41
    )
    kernel = np.ones((1, 1), np.uint8)
    closing = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    img = image_smoothening(img)
    or_image = cv2.bitwise_or(img, filtered)

    return or_image


def get_img(img_path):
    img = cv2.imread(img_path)
    return img


def saveimg(save_path, img):
    cv2.imwrite(save_path, img)


def convert_to_gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def rescale(img):
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    return img


def noise_removal(img):
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


def get_string(img_path):

    img = cv2.imread(img_path)
    img = convert_to_gray(img)
    save_path = r".\gray_img.jpeg"
    cv2.imwrite(save_path, img)

    img = binarize(img)
    save_path = r".\binary_img.jpeg"
    cv2.imwrite(save_path, img)

    img = remove_noise_and_smooth(img)
    save_path = r".\noise_free_img.jpeg"
    cv2.imwrite(save_path, img)


if __name__ == "__main__":
    # img_path = r".\nlp_test.jpeg"
    img_path = r".\ocr.jpeg"
    # img_path = r".\sample_text_handwritten.jpeg"
    print(get_string(img_path))
