import matplotlib.pyplot as plt
import cv2
import easyocr
from pylab import rcParams
from IPython.display import Image

rcParams["figure.figsize"] = 10, 10

reader = easyocr.Reader(["en"])
# Image("C:\Users\shrey\Downloads\Page1.jpeg")
output = reader.readtext(cv2.imread(r"C:\Users\Siddharth\Downloads\stuff.jpeg"))

text = ""
for i in output:
    text += i[1] + " "

print(text)

