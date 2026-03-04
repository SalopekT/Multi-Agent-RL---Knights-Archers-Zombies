import numpy as np
from typing import Any
import pygame
from PIL import Image
import random

def test_examples():
    array = np.load("C:\\Users\\tinsa\\KULeuven\\ml-project-2025-2026-main\\ml-project-2025-2026-main\\observation_data\\example_obs.npy")
    array2 = np.load("C:\\Users\\tinsa\\KULeuven\\ml-project-2025-2026-main\\ml-project-2025-2026-main\\observation_data\\example_zombies.npy")
    print(array.shape)
    print(array2)
    print(array2[0,2])
    print(array2[0,3])

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
    
class DataGenerator:
    def __init__(self):
        self._counter = 0

    #https://docs.ultralytics.com/datasets/#steps-to-contribute-a-new-dataset
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