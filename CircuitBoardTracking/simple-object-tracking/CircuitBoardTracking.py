import cv2
import imutils
import numpy as np




def show(title, img, write = False, wait = False):
    """
    Displays image using OpenCV functions.
    """
    cv2.namedWindow(title, flags = cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)
    cv2.resizeWindow(title, 1200, 900)
    if write:
        cv2.imwrite(title + ".png", img)
    if wait:
        cv2.waitKey(1)

def findRect(im):
    boundingBoxes = []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]

    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    thresh = cv2.threshold(opening, 128, 255, cv2.THRESH_BINARY)[1]
    show("gray", thresh, False, True)
    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in cnts:
        if cv2.contourArea(cnt) < 20000:
            continue
        M = cv2.moments(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        vals = np.array(cv2.mean(im[y:y+h,x:x+w])).astype(np.uint8)
        cv2.rectangle(im, (x,y), (x + w, y + h), (0, 255, 0), 3) 
        cv2.putText(im, str(np.array(cv2.mean(im[y:y+h,x:x+w])).astype(np.uint8)), (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), lineType=cv2.LINE_AA)
        boundingBoxes.append((x, y, x+w, y+h))
        cv2.imshow("frame", im)
        cv2.waitKey(0)
    return boundingBoxes

img = cv2.imread("test.jpg")
im = imutils.resize(img, width=400)
findRect(im)
