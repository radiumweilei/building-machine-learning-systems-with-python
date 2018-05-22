import numpy as np
import mahotas as mh

image = mh.imread('../Dataset/simple-dataset/building05.jpg')
image = mh.colors.rgb2gray(image, dtype=np.uint8)
thresh = mh.thresholding.otsu(image)
print(thresh)
otsubin = (image > thresh)
mh.imsave('../charts/otsu-threshold.jpeg', otsubin.astype(np.uint8) * 255)
otsubin = ~ mh.close(~otsubin, np.ones((15, 15)))
mh.imsave('../charts/otsu-closed.jpeg', otsubin.astype(np.uint8) * 255)

thresh = mh.thresholding.rc(image)
print(thresh)
mh.imsave('../charts/rc-threshold.jpeg', (image > thresh).astype(np.uint8) * 255)
