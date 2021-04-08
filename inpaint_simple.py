# -*- coding:utf-8 –*- #
import cv2
from matplotlib import pyplot as plt
import numpy as np

src_image = cv2.imread("data/1.jpg")
print(src_image.shape)
src_fig = plt.figure()
plt.title("src_image")
plt.imshow(src_image)

gray_image = cv2.cvtColor(src_image, cv2.COLOR_RGB2GRAY)

plt.figure()
plt.title("gray_image")
plt.imshow(gray_image)

# 二值化
ret, binary = cv2.threshold(gray_image, 245, 255, cv2.THRESH_BINARY)
plt.figure()
plt.title("binary")
plt.imshow(binary)

# 进行开操作，去除噪音小点。
kernel_3X3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
binary_after_open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_3X3)

plt.figure()
plt.title("binary_after_open")
plt.imshow(binary_after_open)

restored = cv2.inpaint(src_image, binary_after_open, 9, cv2.INPAINT_NS)
restored_fig = plt.figure()
plt.title("restored")
plt.imshow(restored)

plt.show()
