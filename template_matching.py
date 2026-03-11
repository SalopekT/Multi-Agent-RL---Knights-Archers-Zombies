import cv2
import numpy as np

def remove_clouds(frame):
    mask = cv2.inRange(frame, (60,60,60), (255,255,255))
    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    result = cv2.inpaint(frame, mask, 5, cv2.INPAINT_TELEA)
    return result

#https://docs.opencv.org/3.4/d4/dc6/tutorial_py_template_matching.html
def find_zombies(frame):
    clouds_removed = remove_clouds(frame)
    cv2.imwrite("no_cluds.jpg", clouds_removed)
    gray = cv2.cvtColor(clouds_removed, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('template.png', cv2.IMREAD_GRAYSCALE)
    template = template.squeeze()
    print(template.shape)
    w, h = template.shape[::-1]

    #res = cv2.matchTemplate(gray,template,cv2.TM_CCORR_NORMED)
    res = cv2.matchTemplate(gray,template,cv2.TM_SQDIFF_NORMED)
    threshold = 0.5
    loc = np.where( res <= threshold)
    colored = frame
    for pt in zip(*loc[::-1]):
        cv2.rectangle(colored, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    cv2.imwrite("output.jpg", colored)