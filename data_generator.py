import numpy as np
from typing import Any
import pygame
from PIL import Image, ImageDraw
import random
from ultralytics import YOLO

def test_examples():
    array = np.load("C:\\Users\\tinsa\\KULeuven\\ml-project-2025-2026-main\\ml-project-2025-2026-main\\observation_data\\example_obs.npy")
    array2 = np.load("C:\\Users\\tinsa\\KULeuven\\ml-project-2025-2026-main\\ml-project-2025-2026-main\\observation_data\\example_zombies.npy")
    print(array.shape)
    print(array2)
    print(array2[0,2])
    print(array2[0,3])

#for a global state vector extracts zombie position(typemask => [[1 0 0 0 0 0 x y ...]])
def extract_zombie_coords(all_positions): #all_positions is global game state
    all_zombies = []
    for object in all_positions:
        if object[0]==1:
            x = object[6]
            y = object[7]
            w_bbox = 29.0/1280
            h_bbox = 31.0/720
            zombie_normalized_coords = [x,y,w_bbox,h_bbox]
            all_zombies.append(zombie_normalized_coords)
    return np.array(all_zombies)

#this function is just to draw circles on zombie positions and storing to an image(not really neccessary)
def draw_zombie_positions(env : Any):
    curr_state = env.state()
    zombie_coords = extract_zombie_coords(curr_state)
    image = pygame.surfarray.array3d(env.unwrapped.screen)
    image = np.swapaxes(image,0,1)
    pil_image = Image.fromarray(image,mode='RGB')
    draw = ImageDraw.Draw(pil_image)
    flag = False
    if len(zombie_coords==2):
        flag = True
    for zombie in zombie_coords:
        print(zombie_coords.shape)
        zombie_coords_not_normal = [zombie[0]*1280,zombie[1]*720]
        draw.ellipse([zombie_coords_not_normal[0]-5,zombie_coords_not_normal[1]-5,zombie_coords_not_normal[0]+5,zombie_coords_not_normal[1]+5],fill ="#ffff33", outline ="red")
    if flag == True:
        pil_image.save("zombies_coords_shown.jpg")
        return

#this function draws DETECTED zombie positions
def draw_detected_zombies(env : Any):
    model = YOLO("weights_vision/best(3).pt")
    frame = pygame.surfarray.array3d(env.unwrapped.screen)
    frame = np.swapaxes(frame,0,1)
    pil_image = Image.fromarray(frame,mode='RGB')
    results = model.predict(frame,verbose=False)
    b_boxes = results[0].boxes.xywh
    for b_box in b_boxes:
        real_center_x = b_box[0]+15
        real_center_y = b_box[1]+15

        draw = ImageDraw.Draw(pil_image)
        draw.ellipse([real_center_x-5,real_center_y-5,real_center_x+5,real_center_y+5],fill ="#ffff33", outline ="red")
    #pil_image.save("zombie_detection_shown.jpg")
    


class DataGenerator:
    def __init__(self):
        self._counter = 0

    #https://docs.ultralytics.com/datasets/#steps-to-contribute-a-new-dataset
    #generates for each frame an image and a txt file which describes zombie positions(need this for yolo training)
    def generate_one_data_point(self,env : Any, step):
        #state is a global state
        curr_state = env.state()
        zombie_coords = extract_zombie_coords(curr_state)
        print(zombie_coords)
        #print(curr_state.shape)

        train_or_val = random.randint(1,10)
        if train_or_val > 2:
            for zombie in zombie_coords:
                with open(f"dataset\\labels\\train\\img{self._counter}.txt", "a") as f:
                    f.write(f"0 {zombie[0]} {zombie[1]} {zombie[2]} {zombie[3]}\n")
        else:
            for zombie in zombie_coords:
                with open(f"dataset\\labels\\val\\img{self._counter}.txt", "a") as f:
                    f.write(f"0 {zombie[0]} {zombie[1]} {zombie[2]} {zombie[3]}\n")

        #data is the pixels on the screen
        #https://stackoverflow.com/questions/19982760/get-numpy-array-from-pygame
        data = pygame.surfarray.array3d(env.unwrapped.screen)
        data = np.swapaxes(data,0,1)
        print(data.shape)
        if train_or_val > 2:
            img = Image.fromarray(data, mode='RGB')
            img.save(f"dataset\\images\\train\\img{self._counter}.jpeg")
        else:
            img = Image.fromarray(data, mode='RGB')
            img.save(f"dataset\\images\\val\\img{self._counter}.jpeg")

        self._counter+=1

        raveled_data = data.ravel()
        print(data.shape)
        print(raveled_data.shape)
        flat_obs = raveled_data.astype(np.float32) / 255.0

        

        if (step==0):
            img = Image.fromarray(data, mode='RGB')
            img.save('rgb.png')

def main():
    print("Hello")

if __name__ == "__main__":
    main()