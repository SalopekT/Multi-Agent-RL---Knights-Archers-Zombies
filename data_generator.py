import numpy as np
from typing import Any
import pygame
from PIL import Image, ImageDraw
import random

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
            w_bbox = 29.0/1280
            h_bbox = 31.0/720
            x = object[6]
            y = object[7]
            zombie_normalized_coords = [x,y,w_bbox,h_bbox]
            all_zombies.append(zombie_normalized_coords)
    return np.array(all_zombies)

def extract_own_coordinates(all_positions):
    for object in all_positions:
        if object[5]==1:
            return np.array([object[7],object[8]])

def extract_teammate_coords(all_positions):
    for object in all_positions:
        '''if object[5]==1:
            print(object[7],object[8])'''
        if object[1]==1 and (object[7]!=0.0 or object[8]!=0.0):
            return np.array([object[7],object[8]])

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
def draw_detected_zombies(obs):
    model = YOLO("weights_vision2/best(4).pt")
    '''frame = pygame.surfarray.array3d(env.unwrapped.screen)
    frame = np.swapaxes(frame,0,1)'''
    pil_image = Image.fromarray(obs,mode='RGB')
    results = model.predict(obs,verbose=False)
    b_boxes = results[0].boxes.xywh
    print(len(results[0].boxes))
    for b_box in b_boxes:
        real_center_x = b_box[0]+15
        real_center_y = b_box[1]+15

        draw = ImageDraw.Draw(pil_image)
        draw.ellipse([real_center_x-5,real_center_y-5,real_center_x+5,real_center_y+5],fill ="#ffff33", outline ="red")
    pil_image.save("zombie_detection_shown.jpg")
    


class DataGenerator:
    def __init__(self):
        self._counter = 0
        self.images = []
        self.labels = []

    #generates for each frame an image and a txt file which describes zombie positions(need this for yolo training)
    def generate_one_data_point(self,env : Any, agent, obs, step):
        #state is a global state
        #when calling env.state() in vectorized mode i get global position
        curr_state = env.state()
        #obs = env.state()

        #print(agent_obs)
        zombie_coords = extract_zombie_coords(curr_state)
        #print(zombie_coords)
        #own = extract_own_coordinates(obs)
        #print(own)
        #teammate = extract_teammate_coords(obs)
        #print(teammate)

        train_or_val = random.randint(1,10)
        if train_or_val > 2:
            for zombie in zombie_coords:
                '''pixel_x_diff = 1280*zombie[0]
                pixel_y_diff = 720*zombie[1]
                if (abs(pixel_x_diff)<256 and abs(pixel_y_diff)<256):
                    with open(f"dataset\\labels\\train\\img{self._counter}.txt", "a") as f:
                        f.write(f"0 {zombie[0]} {zombie[1]} {zombie[2]} {zombie[3]}\n")'''
                with open(f"dataset2\\labels\\train\\img{self._counter}.txt", "a") as f:
                        f.write(f"0 {zombie[0]} {zombie[1]} {zombie[2]} {zombie[3]}\n")
            '''if teammate is not None:
                pixel_x_diff = 1280*teammate[0]
                pixel_y_diff = 720*teammate[1]
                if abs(pixel_x_diff)<256 and abs(pixel_y_diff) < 256:
                    with open(f"dataset\\labels\\train\\img{self._counter}.txt", "a") as f:
                            f.write(f"1 {teammate[0]} {teammate[1]} 0.03 0.03\n")
            with open(f"dataset\\labels\\train\\img{self._counter}.txt", "a") as f:
                            f.write(f"1 0 0 0.03 0.03\n")'''
        else:
            for zombie in zombie_coords:
                pixel_x_diff = 1280*zombie[0]
                pixel_y_diff = 720*zombie[1]
                '''if (abs(pixel_x_diff)<256 and abs(pixel_y_diff)<256):
                    with open(f"dataset\\labels\\val\\img{self._counter}.txt", "a") as f:
                        f.write(f"0 {zombie[0]} {zombie[1]} {zombie[2]} {zombie[3]}\n")'''
                with open(f"dataset2\\labels\\val\\img{self._counter}.txt", "a") as f:
                        f.write(f"0 {zombie[0]} {zombie[1]} {zombie[2]} {zombie[3]}\n")
            '''if teammate is not None:
                pixel_x_diff = 1280*teammate[0]
                pixel_y_diff = 720*teammate[1]
                if abs(pixel_x_diff)<256 and abs(pixel_y_diff) < 256:
                    with open(f"dataset\\labels\\val\\img{self._counter}.txt", "a") as f:
                        f.write(f"1 {teammate[0]} {teammate[1]} 0.03 0.03\n")
            with open(f"dataset\\labels\\val\\img{self._counter}.txt", "a") as f:
                            f.write(f"1 0 0 0.03 0.03\n")'''

        #data is the pixels on the screen
        #https://stackoverflow.com/questions/19982760/get-numpy-array-from-pygame

        ##
        ##if i continue like this i need to crop full  screen to generate training images
        ##
        data = pygame.surfarray.array3d(env.unwrapped.screen)
        data = np.swapaxes(data,0,1)
        img = Image.fromarray(data,mode = 'RGB')
        if train_or_val > 2:
            img.save(f"dataset2\\images\\train\\img{self._counter}.jpeg")
        else:
            img.save(f"dataset2\\images\\val\\img{self._counter}.jpeg")
        '''print(data.shape)
        own_pixel_x = int(own[0] * 1280)
        own_pixel_y = int(own[1] * 720)

        left   = own_pixel_x - 256
        right  = own_pixel_x + 256
        top    = own_pixel_y - 256
        bottom = own_pixel_y + 256

        crop_left  = max(0, left)
        crop_top   = max(0, top)
        crop_right = min(1280, right)
        crop_bottom= min(720, bottom)

        paste_x = max(0, -left)
        paste_y = max(0, -top)

        black_img = np.zeros((512, 512, 3), dtype=np.uint8)

        valid = data[crop_top:crop_bottom, crop_left:crop_right]

        h, w = valid.shape[:2]

        # paste it
        black_img[paste_y:paste_y+h, paste_x:paste_x+w] = valid

        if train_or_val > 2:
            img = Image.fromarray(black_img, mode='RGB')
            img.save(f"dataset\\images\\train\\img{self._counter}.jpeg")
        else:
            img = Image.fromarray(black_img, mode='RGB')
            img.save(f"dataset\\images\\val\\img{self._counter}.jpeg")'''

        self._counter+=6

        raveled_data = data.ravel()
        print(data.shape)
        print(raveled_data.shape)
        flat_obs = raveled_data.astype(np.float32) / 255.0

        

        if (step==0):
            img.save('rgb.png')
    
    '''def generate_one_data_point(self,env : Any, agent, obs, step):
        zombie_coords = extract_zombie_coords(obs)
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
        if (step==0):
            img = Image.fromarray(data, mode='RGB')
            img.save('rgb.png')'''
    
    def generate_angle_data(self,env : Any, obs, step):
         print(self._counter)
         if self._counter==5000:
             images = np.array(self.images)
             labels = np.array(self.labels)
             np.savez("dataset_angles/dataset_test2.npz", images=images, labels=labels)
         data_own = []
         for object in obs:
             if np.isscalar(object[5]) and object[5]==1:
                 x_own = object[7]
                 y_own = object[8]
                 x_own = int(x_own * 1280)+15
                 y_own = int(y_own * 720)+15
                 x_heading = object[9]
                 y_heading = object[10]
                 data_own = [x_own,y_own,x_heading,y_heading]

                 data_full_screen = pygame.surfarray.array3d(env.unwrapped.screen)
                 data_full_screen = np.swapaxes(data_full_screen,0,1)
                 
                 heading = [x_heading,y_heading]
                 self.labels.append(heading)
                 x_min = x_own - 20
                 x_max = x_own + 21
                 y_min = y_own - 20
                 y_max = y_own + 21

                 x_min = max(0, x_min)
                 x_max = min(1280, x_max)
                 y_min = max(0, y_min)
                 y_max = min(720, y_max)

                 crop = data_full_screen[y_min:y_max, x_min:x_max, :]
                 h, w, _ = crop.shape

                 pad_h = max(0, 20 - h)
                 pad_w = max(0, 20 - w)

                 pad_top = pad_h // 2
                 pad_bottom = pad_h - pad_top

                 pad_left = pad_w // 2
                 pad_right = pad_w - pad_left
                 crop = np.pad(
                    crop,
                    ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                    mode='constant',
                    constant_values=0
                )
                 self.images.append(crop)
                 img = Image.fromarray(crop.astype(np.uint8))
                 img.save("test_crop.png")

         self._counter+=1
         return data_own
         

def main():
    print("Hello")

if __name__ == "__main__":
    main()