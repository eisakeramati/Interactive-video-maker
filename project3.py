import numpy as np
import cv2
import random


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


img = cv2.imread("hope.png")
img1 = image_resize(img, height=15, width=15)
height = img1.shape[0]
width = img1.shape[1]
temp2 = np.array([0, 0, 0])
temp = np.array([255, 255, 255])


def overlayer(start1, start2, image, mask):
    res = []
    out = []
    for i in range(0, len(start1)):
        out.append(True)
    for t in range(0, len(start1)):
        res.append(False)
    for p in range(0, len(start2)):
        if ((start1[p] + img1.shape[0]) < mask.shape[0]):
            if (mask[start1[p] + img1.shape[0], start2[p]] != temp2).all():
                res[p] = True
            height = img1.shape[0]
            width = img1.shape[1]
            for i in range(0, height):
                for j in range(0, width):
                    if (img1[i, j] != temp).all():
                        image[start1[p] + i, start2[p] + j] = img1[i, j]
        else:
            out[p] = False
    return res, image, out


vid = cv2.VideoCapture(0)
# subtractor = cv2.createBackgroundSubtractorMOG2()
subtractor = cv2.createBackgroundSubtractorKNN()
s = []
for i in range(0, 20):
    s.append(0)
sarr = []
fourcc = cv2.VideoWriter_fourcc(*'XVID')

ret, frame = vid.read()
row, col, ch = frame.shape
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (col, row))

for i in range(0, 20):
    sarr.append(random.randint(1, frame.shape[1] - img1.shape[0]))
print((frame.shape[1]) / 2)
print((frame.shape[1]))
while (frame is not None):
    fgmask = subtractor.apply(frame)

    morph = fgmask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    ret, thresh3 = cv2.threshold(morph, 100, 255, cv2.THRESH_BINARY)
    fgmask = thresh3

    bool, fgmask2, check = overlayer(s, sarr, frame, fgmask)
    cv2.imshow('frame', fgmask2)
    out.write(frame)
    cv2.imshow('mask', fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    for i in range(0, len(bool)):
        if bool[i] == False:
            s[i] = s[i] + 8
    ret, frame = vid.read()
    for i in range(0, len(check)):
        if check[i] == False:
            s[i] = 0
            sarr[i] = random.randint(1, frame.shape[1] - img1.shape[0])
vid.release()
out.release()
cv2.destroyAllWindows()
