"""Template of your submission file for Task 3 (multi agent KAZ).
"""
from typing import Callable
import gymnasium
import time
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType
from PIL import Image
import template_matching as tm
from pathlib import Path
from ray.rllib.core.rl_module import MultiRLModule
import torch
import numpy as np
import cv2
from NeuralNet.AngleNet import AngleNet
from gymnasium import spaces
import os
import sys
import pygame
import math

current_dir = os.path.dirname(os.path.abspath(__file__))
yolo_path = os.path.join(current_dir, "pytorch-YOLOv4")

sys.path.append(yolo_path)

from tool.darknet2pytorch import Darknet

class Leniency:
    def __init__(self, N):
        self.leniency = N
        self.last_N_rewards = {0:[],1:[],2:[],3:[],4:[],5:[]}

    def add_reward(self,reward,move):
        if len(self._last_N_rewards[move]) < self.leniency:
            self._last_N_rewards[move].append(reward)

    def clear_rewards(self,move):
         self._last_N_rewards[move].clear()
        

class CustomWrapper(BaseWrapper):
    """
    Wrapper to use to add state pre-processing (feature engineering)
    """
    def __init__(self, env):
        super().__init__(env)
        package_directory = os.path.dirname(os.path.abspath(__file__))
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = torch.device("cpu")
        angle_model_path = os.path.join(package_directory, "NeuralNet", "anglenet.pth")

        # load the model
        state_dict = torch.load(angle_model_path, map_location="cpu")
        self.model = AngleNet().to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.zombie_detector = CustomZombieDetectorFunction(self.env)
        self.zombie_tick=0
        self.cached_zombies = []

        self.angle_archer0 = 0
        self.angle_archer1 = 0
        self.archer0_direction = pygame.Vector2(0, -1)
        self.archer1_direction = pygame.Vector2(0, -1)
        self.ang_rate = 10
        



    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return spaces.Box(low = -1.0, high = 1.0, shape = (3*2+5*3,))

    def observe(self, agent: AgentID) -> ObsType | None:
        #start = time.perf_counter()
        
        '''screen = pygame.surfarray.array3d(self.env.unwrapped.screen)
        screen= np.swapaxes(screen,0,1)'''

        obs = self.env.observe(agent)
        state = self.env.state()

        own_pos = [0,0]
        teammate_pos = [0,0]
        archers = list(self.env.unwrapped.archer_list)
        '''if len(archers)==2:
            if (agent == "archer_0"):
                own_pos = archers[0].rect.center
                teammate_pos = archers[1].rect.center
            else:
                own_pos = archers[1].rect.center
                teammate_pos = archers[0].rect.center'''
               
        
        x, y = own_pos
        teammate_pos_rel_x = teammate_pos[0]-x
        teammate_pos_rel_y = teammate_pos[1]-y

        x_min = max(0,x-20)
        y_min = max(0,y-20)
        x_max = min(1280,x+21)
        y_max = min(720,y+21)
                
        crop = obs[y_min:y_max, x_min:x_max, :]
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
            output_heading = self.model(crop_tensor)
            output_heading = output_heading.squeeze(0)
        
        #print(output_heading[0].item(), output_heading[1].item())
        

        zombies = self.zombie_detector(obs)
        num_zombies = len(zombies)
        sorted_zombies = sorted(zombies, key=lambda z: z[1])
        #print("----------")
        sorted_zombies.reverse()
        #print(sorted_zombies)
        #print(f"Zombie detection took {end-start:.6f} seconds")
        
        #in final obs i normalize to get values from -1 to 1, before normalizing they are either -h to h (x) or -w to w(y)
        if agent == "archer_0":
            output_heading = self.archer0_direction
        elif agent == "archer_1":
            output_heading = self.archer1_direction
        final_obs = [x/1280,y/720,
                        #output_heading[0].item(),output_heading[1].item(),
                         output_heading[0], output_heading[1],
                         teammate_pos_rel_x/1280,teammate_pos_rel_y/720]
        #print(self.angle_archer0)
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

        final_obs = np.clip(final_obs, -1.0, 1.0)
        #print(final_obs)
        #end = time.perf_counter()
        #print(f"Zombie detection took {end-start:.6f} seconds")
        return np.array(final_obs, dtype=np.float32)
   
                
    def step(self, action):
        agent = self.env.agent_selection
        if action==2: 
            if agent == "archer_0":
                
                self.angle_archer0 += self.ang_rate
                self.archer0_direction = pygame.Vector2(0, -1).rotate(-self.angle_archer0)
            else:
                
                self.angle_archer1 += self.ang_rate
                self.archer1_direction = pygame.Vector2(0, -1).rotate(-self.angle_archer1)
        elif action==3:
            if agent == "archer_0":
                
                self.angle_archer0 -= self.ang_rate
                self.archer0_direction = pygame.Vector2(0, -1).rotate(-self.angle_archer0)
            else:
                
                self.angle_archer1 -= self.ang_rate
                self.archer1_direction = pygame.Vector2(0, -1).rotate(-self.angle_archer1)
    
        print(self.angle_archer0)
        return self.env.step(action)
            
    def reset(self, seed=None, options=None):
        self.angle_archer0 = 0
        self.angle_archer1 = 0

        self.archer0_direction = pygame.Vector2(0, -1)
        self.archer1_direction = pygame.Vector2(0, -1)

        self.zombie_tick = 0
        self.cached_zombies = []

        return self.env.reset(seed=seed, options=options)

            


class CustomPredictFunction(Callable):
    """Function to use to load the trained model and predict the action"""

    def __init__(self, env):

        # Here you should load your trained model(s) from a checkpoint in your folder
        best_checkpoint = (Path("results4") / "learner_group" / "learner" / "rl_module").resolve()
        self.modules = MultiRLModule.from_checkpoint(best_checkpoint)
        self.archer0_heading = 0
        self.archer0_direction = pygame.Vector2(0, -1)
        self.archer1_direction = pygame.Vector2(0, -1)
        self.archer1_heading = 0
        self.ang_rate = 10
        print("initialized ppo predictor")
        #self.policy = self.modules["shared_policy"]

    def __call__(self, observation, agent, *args, **kwargs):
        
        rl_module = self.modules[agent]
        fwd_ins = {"obs": torch.Tensor(observation).unsqueeze(0)}
        #fwd_ins = observation
        fwd_outputs = rl_module.forward_inference(fwd_ins)
        action_dist_class = rl_module.get_inference_action_dist_cls()
        action_dist = action_dist_class.from_logits(
            fwd_outputs["action_dist_inputs"]
        )
        action = action_dist.sample()[0].numpy()
       
        return action
        


class CustomZombieDetectorFunction(Callable):
    """Function to use to load the trained model and predict where
    the zombies are.
    """
        
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
        #self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        print(os.path.abspath(weights_path))
        checkpoint = torch.load(weights_path, map_location=self.device)
        print(type(checkpoint))
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict, strict=False)
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
        img = observation.copy()
        for box in boxes_new:
            box = box.tolist()
            x,y,w,h = box

            x = int(x*1280)
            y = int(y*720)
            w = int(w*1280)
            h = int(h*720)
            if x+15<1280 and y+15<720:
                zombie = [x+30,y+30,30,30]
            else:
                zombie = [x,y,30,30]
            matrix.append(zombie)
            # Top-left and bottom-right corners
            pt1 = (x, y)
            pt2 = (x + w, y + h)

            # Draw rectangle (color = green, thickness = 2)
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
        #print(matrix)
        cv2.imwrite("debug_output.png", img)
        

        return matrix
        

       
        

