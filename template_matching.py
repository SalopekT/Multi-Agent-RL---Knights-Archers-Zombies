import cv2
import numpy as np

'''def remove_stars(frame):
    opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)'''

def remove_clouds(frame):
    mask = cv2.inRange(frame, (60,60,60), (255,255,255))
    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    result = cv2.inpaint(frame, mask, 5, cv2.INPAINT_TELEA)
    return result

#https://docs.opencv.org/3.4/d4/dc6/tutorial_py_template_matching.html
#https://thepythoncode.com/article/non-maximum-suppression-using-opencv-in-python
def find_zombies(frame):
    '''clouds_removed = remove_clouds(frame)
    cv2.imwrite("no_cluds.jpg", clouds_removed)
    gray = cv2.cvtColor(clouds_removed, cv2.COLOR_BGR2GRAY)'''
    frame_width = len(frame[0])
    frame = np.array(frame)          # keep original type
    print("lalala")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('even_smaller_template.png', cv2.IMREAD_GRAYSCALE)
    template = template.squeeze()
    print(template.shape)
    w, h = template.shape[::-1]

    #res = cv2.matchTemplate(gray,template,cv2.TM_CCORR_NORMED)
    res = cv2.matchTemplate(gray,template,cv2.TM_SQDIFF_NORMED)
    threshold = 0.45
    loc = np.where( res <= threshold)
    colored = frame.copy()
    boxes = []

    for pt in zip(*loc[::-1]): #this creates pairs (x,y)
        boxes.append([pt[0], pt[1], w, h])

        #cv2.rectangle(colored, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

    confidence_scores = []
    for pt in boxes:
        confidence_scores.append(1-res[pt[1],pt[0]])

    indices = cv2.dnn.NMSBoxes(bboxes=boxes, scores=confidence_scores, score_threshold=0.4, nms_threshold=0.2)
    filtered_boxes = [boxes[i] for i in indices]

    for pt in filtered_boxes:
        x,y,w,h = pt
        '''if (x>frame_width-50 or x<50):
            continue'''
        if (x+w>0 and x+w<512 and y+h>0 and y+h<512):
            cv2.rectangle(colored, (x,y), (x + w, y + h), (0,0,255), 2)
    cv2.imwrite("output.jpg", colored)
    return filtered_boxes, indices


def observations_matching(obs, state): #finds where in the image is a player
    gray_state = cv2.cvtColor(state.astype('uint8'), cv2.COLOR_BGR2GRAY)
    gray_obs   = cv2.cvtColor(obs.astype('uint8'), cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(gray_state, gray_obs, cv2.TM_SQDIFF)
    height, width = obs.shape[:2]
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return [min_loc[0]+256,min_loc[1]+256]
