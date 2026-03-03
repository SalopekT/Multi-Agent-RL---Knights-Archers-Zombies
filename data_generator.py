import numpy as np
from typing import Any
import pygame
from PIL import Image

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
            w = 29
            h = 31
            zombie_normalized_coords = [x,y]
            all_zombies.append(zombie_normalized_coords)
    return np.array(all_zombies)
    

def generate_one_data_point(env : Any, step):
    #state is a global state
    curr_state = env.state()
    zombie_coords = extract_zombie_coords(curr_state)
    print(zombie_coords)
    #print(curr_state.shape)

    #data is the pixels on the screen
    #https://stackoverflow.com/questions/19982760/get-numpy-array-from-pygame
    data = pygame.surfarray.array3d(env.unwrapped.screen)
    data = np.swapaxes(data,0,1)
    raveled_data = data.ravel()
    print(raveled_data.shape)
    flat_obs = raveled_data.astype(np.float32) / 255.0

    

    if (step==0):
        img = Image.fromarray(data, mode='RGB')
        img.save('rgb.png')

def main():
    print("Hello")

if __name__ == "__main__":
    main()