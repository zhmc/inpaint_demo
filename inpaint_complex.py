# -*- coding:utf-8 –*- #
import cv2
from matplotlib import pyplot as plt
import numpy as np

src_image = cv2.imread("data/2.jpg")
# src_image = src_image[100:150, 0:50, :].copy()
src_fig = plt.figure()
plt.subplot(121)
plt.title("src_image")
plt.imshow(src_image)

print(src_image.shape)

gray_image = cv2.cvtColor(src_image, cv2.COLOR_RGB2GRAY)
plt.subplot(122)
plt.title("gray_image")
plt.imshow(gray_image)

# 观察灰度图，发现网格线位于图像中的谷线

# Do some rough edge detection to find the grid
# 对X轴方向和Y轴方向做Sobel梯度提取
sX = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)
sY = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)

# 去除负值。这是为了提取单边界，否则网格线两边的梯度绝对值都较大，形成双边缘。
sX[sX < 0] = 0
sY[sY < 0] = 0

plt.figure()
plt.subplot(221)
plt.title("Sobel X")
plt.imshow(sX)

plt.subplot(222)
plt.title("Sobel Y")
plt.imshow(sY)

# the sum operation projects the edges to the X or Y-axis.
# The 0.2 damps the high peaks a little
# 对sobel的值进行exp(0.2), 0.2会稍微减弱峰值。
# 然后进行竖直方向的积分投影
eX = (sX ** .2).sum(axis=0)
print(eX.shape)
eX = np.roll(eX, -1)  # correct for the 1-pixel offset due to Sobel filtering
# 将eX的值向前滚动一位。因为Sobel算子计算时，对于单线的局部最大值，会在两边计算出
# 边缘。当前面过滤掉负值后，右边的边缘与实际的局部最大值相差一个像素。
# 此处是基于网格线宽度为一个像素的场景特征。
print(eX.shape)
plt.subplot(223)
plt.plot(eX)

# 对sobel的值进行exp(0.2), 0.2会稍微减弱峰值。
# 然后进行水平方向的积分投影
eY = (sY ** .2).sum(axis=1)
eY = np.roll(eY, -1)
plt.subplot(224)
plt.plot(eY)

plt.figure()
# plt.subplot(121)
plt.imshow(src_image)
plt.title("src_image")

# 构建掩膜
A2 = src_image.copy()
mask = np.zeros(A2.shape[:2], dtype=np.uint8)
mask[eY > 480, :] = 1
mask[:, eX > 390] = 1

A2[mask.astype(bool), :] = 255

# plt.subplot(222)
plt.figure()
plt.imshow(A2)
plt.title("mask")

restored = cv2.inpaint(src_image, mask, 1, cv2.INPAINT_NS)

# plt.subplot(122)
plt.figure()
plt.imshow(restored)
plt.title("restored")

plt.show()
