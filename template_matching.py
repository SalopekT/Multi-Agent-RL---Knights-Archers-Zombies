import cv2
import numpy as np

#https://docs.opencv.org/3.4/d4/dc6/tutorial_py_template_matching.html
def find_zombies(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('template.png', cv2.IMREAD_GRAYSCALE)
    template = template.squeeze()
    print(template.shape)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.4
    loc = np.where( res >= threshold)
    colored = frame
    for pt in zip(*loc[::-1]):
        cv2.rectangle(colored, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    cv2.imwrite("output.jpg", colored)