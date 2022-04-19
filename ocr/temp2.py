import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
cv2.namedWindow("output", cv2.WINDOW_NORMAL)
# Grayscale, Gaussian blur, Otsu's threshold
image = cv2.imread("test.jpeg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3, 3), 0)
# thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Morph open to remove noise and invert image
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
# invert = 255 - opening
invert = 255 - thresh
invertS = cv2.resize(invert, (720, 1080))
# Perform text extraction
data = pytesseract.image_to_string(invert, lang="eng", config="--psm 6")
print(data)

threshS = cv2.resize(thresh, (720, 1280))
cv2.imshow("thresh", threshS)
# cv2.imshow("opening", opening)
cv2.imshow("invert", invertS)
cv2.waitKey()
