"""
This file contains an example of implementation of the CustomWapper and CustomPredictFunction that you need to submit.

Here, we are using Ray RLLib to load the trained agents.
"""

from pathlib import Path
import random
from typing import Optional
from typing import Callable
import numpy as np
from PIL import Image
import torch
from gymnasium import spaces
import gymnasium
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType
from ray.rllib.core.rl_module import MultiRLModule
import pygame
import time
from NeuralNet.AngleNet import AngleNet
import os
import sys
import cv2
sys.path.append("C:\\Users\\tinsa\\KULeuven\\ml-project-2025-2026-main\\ml-project-2025-2026-main\\pytorch-YOLOv4")
from tool.darknet2pytorch import Darknet

class CustomWrapper(BaseWrapper):

    def __init__(self, env, target_size=(64, 64)):
        super().__init__(env)
        self.target_size = target_size  # (H, W)
        package_directory = os.path.dirname(os.path.abspath(__file__))
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = torch.device("cpu")
        angle_model_path = os.path.join(package_directory, "NeuralNet", "anglenet.pth")

        # load the model
        self.model = AngleNet().to(self.device)
        self.model.load_state_dict(torch.load(angle_model_path, map_location=self.device))
        self.model.eval()

        self.zombie_detector = CustomZombieDetectorFunction(self.env)

    def observation_space(self, agent: AgentID):
        return spaces.Box(low = -1.0, high = 1.0, shape = (3*2+5*3,))

    def observe(self, agent: AgentID) -> ObsType | None:
        start = time.perf_counter()
        obs = super().observe(agent)
        screen = pygame.surfarray.array3d(self.unwrapped.screen)
        screen = np.swapaxes(screen, 0, 1)

        screen = screen.astype(np.uint8)
        state = super().state()
        #print(state)
        own_pos = [0,0]
        teammate_pos = [0,0]
        archers = list(self.env.unwrapped.archer_list)
        if len(archers)==2:
            if (agent == "archer_0"):
                own_pos = archers[0].rect.center
                teammate_pos = archers[1].rect.center
            else:
                own_pos = archers[1].rect.center
                teammate_pos = archers[0].rect.center
               
        
        x, y = own_pos
        teammate_pos_rel_x = teammate_pos[0]-x
        teammate_pos_rel_y = teammate_pos[1]-y

        x_min = max(0,x-20)
        y_min = max(0,y-20)
        x_max = min(1280,x+21)
        y_max = min(720,y+21)
                
        crop = screen[y_min:y_max, x_min:x_max, :]
        h, w, c = crop.shape

        to_pad_bottom,to_pad_top,to_pad_right,to_pad_left=0,0,0,0
        if h<41:
            to_pad = 41-h
            to_pad_bottom = to_pad//2
            to_pad_top = to_pad-to_pad_bottom

        if w<41:
            to_pad = 41-w
            to_pad_right = to_pad//2
            to_pad_left = to_pad-to_pad_right

        if h<41 or w <41:
            crop = np.pad(crop,((to_pad_bottom, to_pad_top),(to_pad_left, to_pad_right),(0,0)),mode='constant',constant_values=0)
        
        
        self.model.eval()
        crop_tensor = ((torch.from_numpy(crop).float()/255.0)-0.5)*2
        crop_tensor = crop_tensor.permute(2, 0, 1)
        crop_tensor = crop_tensor.unsqueeze(0) 
        
        with torch.no_grad():
            output_heading1 = self.model(crop_tensor)
            output_heading1 = output_heading1.squeeze(0)
        print([output_heading1[0].item(), output_heading1[1].item()])
        zombies = self.zombie_detector(screen)
        sorted_zombies = sorted(zombies, key=lambda z: z[1])
        #print("----------")
        sorted_zombies.reverse()
        #print(sorted_zombies)
        output_heading = []
        for object in obs:
             if np.isscalar(object[5]) and object[5]==1:
                 x_own = object[7]
                 y_own = object[8]
                 x_own = int(x_own * 1280)+15
                 y_own = int(y_own * 720)+15
                 x_heading = object[9]
                 y_heading = object[10]
                 data_own = [x_own,y_own,x_heading,y_heading]
                 output_heading = [x_heading,y_heading]
        if len(output_heading)!=2:
            output_heading = [-1,0]
        print(output_heading)
        zombies = []
        for object in state:
            if object[0]==1:
                w_bbox = 29.0/1280
                h_bbox = 31.0/720
                x = int(object[6]*1280)
                y = int(object[7]*720)
                zombie_pos = [x,y]
                zombies.append(zombie_pos)
        num_zombies = len(zombies)
        #print("Zombies: ")
        #print(zombies)
        zombie_y_values = []
        for i in range(num_zombies):
            zombie_rel_x = zombies[i][0] - x
            zombie_rel_y = zombies[i][1] - y
            zombie_y_values.append(zombies[i][0])
        sorted_zombies = sorted(zombies, key=lambda z: z[1])
        #print("----------")
        sorted_zombies.reverse()
        #print(sorted_zombies)
        print("----------------")
        #print("Sorted zombies: ")
        #print(sorted_zombies)
        final_obs = [x/1280,y/720,
                         #output_heading[0],output_heading[1],
                         teammate_pos_rel_x/1280,teammate_pos_rel_y/720]
        if num_zombies<5:
            for i in range(num_zombies):
               zombie_rel_x = sorted_zombies[i][0] - x
               zombie_rel_y = sorted_zombies[i][1] - y
               final_obs.extend([zombie_rel_x/1280,zombie_rel_y/720,1.0])
            for i in range(5-num_zombies):
                final_obs.extend([0.0,0.0,0.0])
        if num_zombies >= 5:
            for i in range(5):
               zombie_rel_x = sorted_zombies[i][0] - x
               zombie_rel_y = sorted_zombies[i][1] - y
               final_obs.extend([zombie_rel_x/1280,zombie_rel_y/720,1.0])
        final_obs = np.array(final_obs, dtype=np.float32)
        #print(final_obs)
        end = time.perf_counter()
        print(f"Zombie detection took {end-start:.6f} seconds")
        return final_obs


class CustomPredictFunction(Callable):
    """ This is an example of an instantiation of the CustomPredictFunction that loads a trained RLLib algorithm from
    a checkpoint and extract the policies from it"""

    def __init__(self, env):

        # Here you should load your trained model(s) from a checkpoint in your folder
        best_checkpoint = (Path("results") / "learner_group" / "learner" / "rl_module").resolve()
        self.modules = MultiRLModule.from_checkpoint(best_checkpoint)

        self.archer0_heading = 0
        self.archer0_direction = pygame.Vector2(0, -1)
        self.archer1_direction = pygame.Vector2(0, -1)
        self.archer1_heading = 0
        self.ang_rate = 10

    def __call__(self, observation, agent, *args, **kwargs):
        '''rl_module = self.modules[agent]
        fwd_ins = {"obs": torch.Tensor(observation).unsqueeze(0)}
        fwd_outputs = rl_module.forward_inference(fwd_ins)
        action_dist_class = rl_module.get_inference_action_dist_cls()
        action_dist = action_dist_class.from_logits(
            fwd_outputs["action_dist_inputs"]
        )
        action = action_dist.sample()[0].numpy()
        return action'''
        #here i can insert real heading
        if agent == "archer_0":
            new_obs = np.insert(observation, 2, self.archer0_direction[0])
            new_obs = np.insert(new_obs, 3, self.archer0_direction[1])
        else:
            new_obs = np.insert(observation, 2, self.archer1_direction[0])
            new_obs = np.insert(new_obs, 3, self.archer1_direction[1])
        #print(new_obs)
        rl_module = self.modules[agent]
        fwd_ins = {"obs": torch.Tensor(new_obs).unsqueeze(0)}
        #fwd_ins = observation
        fwd_outputs = rl_module.forward_inference(fwd_ins)
        action_dist_class = rl_module.get_inference_action_dist_cls()
        action_dist = action_dist_class.from_logits(
            fwd_outputs["action_dist_inputs"]
        )
        action = action_dist.sample()[0].numpy()
        if action==2: 
            if agent == "archer_0":
                self.archer0_heading += self.ang_rate
                self.archer0_direction = pygame.Vector2(0, -1).rotate(-self.archer0_heading)
            else:
                self.archer1_heading += self.ang_rate
                self.archer1_direction = pygame.Vector2(0, -1).rotate(-self.archer1_heading)
        elif action==3:
            if agent == "archer_0":
                self.archer0_heading -= self.ang_rate
                self.archer0_direction = pygame.Vector2(0, -1).rotate(-self.archer0_heading)
            else:
                self.archer1_heading -= self.ang_rate
                self.archer1_direction = pygame.Vector2(0, -1).rotate(-self.archer1_heading)
       
        
        #print(self.archer0_direction)
        #print(action)
        return action


class CustomZombieDetectorFunction(Callable):
    def __init__(self, env: gymnasium.Env):
        #cv2.setNumThreads(0)
        package_directory = os.path.dirname(os.path.abspath(__file__))

        # construct absolute paths to YOLO weights and cfg
        weights_path = os.path.join(package_directory, "yolov4-tiny-weights", "my_yolov4-tiny.pth")
        cfg_path = os.path.join(package_directory, "yolov4-tiny-obj.cfg")

        self.model = Darknet(cfg_path)
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        #self.device = torch.device("cuda")
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        

    def __call__(self, observation, *args, **kwargs):
        """Returns a matrix of shape (nb_zombies, nb_attributes), where
        the attributes are defining a rectangle with (x,y,width,heigh) and
        indicate where the zombies are. The zombies are ordered from most
        likely to least likely positions. The evaluation uses the first k
        items if there are k zombies on the screen.
        """
        
        matrix = []
        img = cv2.resize(observation, (416, 416))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).float() / 255.0  
        img = img.permute(2, 0, 1).unsqueeze(0).to(self.device) 

        with torch.inference_mode():
            outputs = self.model(img)

        #print(outputs)
        boxes, confidences = outputs
        boxes = boxes.squeeze(0).squeeze(1)  
        confidences = confidences.squeeze(0).squeeze(1)

        '''boxes_new = []
        for i in range(len(confidences)):
            if confidences[i]>0.9:
                boxes_new.append(boxes[i])'''
        mask = confidences > 0.9
        boxes_new = boxes[mask]
        
        #print(max(confidences))

        for box in boxes_new:
            box = box.tolist()
            x,y,w,h = box

            x = int(x*1280)
            y = int(y*720)
            w = int(w*1280)
            h = int(h*720)
            if x+15<1280 and y+15<720:
                zombie = [x+15,y+15,30,30]
            else:
                zombie = [x,y,30,30]
            matrix.append(zombie)
        #print(matrix)
        img = observation.copy()

        return matrix

