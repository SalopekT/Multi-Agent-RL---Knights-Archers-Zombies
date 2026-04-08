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
sys.path.append("C:\\Users\\tinsa\\KULeuven\\ml-project-2025-2026-main\\ml-project-2025-2026-main\\pytorch-YOLOv4")
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
        cv2.setNumThreads(0)
        self.device = torch.device("cpu")
        angle_model_path = os.path.join(package_directory, "NeuralNet", "anglenet.pth")

        # load the model
        self.model = AngleNet().to(self.device)
        self.model.load_state_dict(torch.load(angle_model_path, map_location=self.device))
        self.model.eval()

        self.zombie_detector = CustomZombieDetectorFunction(self.env)

        self.leniency_archer_0 = Leniency(10)
        self.leniency_archer_1 = Leniency(10)



    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return spaces.Box(low = -1.0, high = 1.0, shape = (3*2+5*3,))

    def observe(self, agent: AgentID) -> ObsType | None:
        max_zombies = self.env.unwrapped.max_zombies
        #obs = self.env.unwrapped.observe(agent)
        obs = self.env.observe(agent)
        state = self.env.state()

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
            output_infer = self.model(crop_tensor)
            output_infer = output_infer.squeeze(0)
        
        #start = time.perf_counter()
        zombies = self.zombie_detector(obs)
        #end = time.perf_counter()
        #print(f"Zombie detection took {end-start:.6f} seconds")
        
        num_zombies = len(zombies)
        #in final obs i normalize to get values from -1 to 1, before normalizing they are either -h to h (x) or -w to w(y)
        final_obs = [x/1280,y/720,
                         output_infer[0].item(),output_infer[1].item(),
                         teammate_pos_rel_x/1280,teammate_pos_rel_y/720]
        if num_zombies<5:
            for i in range(num_zombies):
               zombie_rel_x = zombies[i][0] - x
               zombie_rel_y = zombies[i][1] - y
               final_obs.extend([zombie_rel_x/1280,zombie_rel_y/720,1.0])
            for i in range(5-num_zombies):
                final_obs.extend([0.0,0.0,-1.0])
        if num_zombies >= 5:
            for i in range(5):
               zombie_rel_x = zombies[i][0] - x
               zombie_rel_y = zombies[i][1] - y
               final_obs.extend([zombie_rel_x/1280,zombie_rel_y/720,1.0])
        #print(final_obs)
        final_obs = np.array(final_obs, dtype=np.float32)

        final_obs = np.clip(final_obs, -1.0, 1.0)
        return np.array(final_obs, dtype=np.float32)
    
    def step(self, action):
        curr_agent = self.env.agent_selection
        self.env.step(action)
        obs, reward, termination, truncation, info = self.last()

        if (curr_agent == "archer_0"):
            self.leniency_archer_0.add_reward(reward)
        


            


class CustomPredictFunction(Callable):
    """Function to use to load the trained model and predict the action"""

    def __init__(self, env):

        # Here you should load your trained model(s) from a checkpoint in your folder
        best_checkpoint = (Path("results") / "learner_group" / "learner" / "rl_module").resolve()
        self.modules = MultiRLModule.from_checkpoint(best_checkpoint)

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
    
    '''def __init__(self, env):
        self.env = env

    def __call__(self, observation, agent, *args, **kwargs):
        return self.env.action_space(agent).sample()'''


class CustomZombieDetectorFunction(Callable):
    """Function to use to load the trained model and predict where
    the zombies are.
    """
        
    def __init__(self, env: gymnasium.Env):
        cv2.setNumThreads(0)
        package_directory = os.path.dirname(os.path.abspath(__file__))

        # construct absolute paths to YOLO weights and cfg
        weights_path = os.path.join(package_directory, "yolov4-tiny-weights", "my_yolov4-tiny.pth")
        cfg_path = os.path.join(package_directory, "yolov4-tiny-obj.cfg")

        self.model = Darknet(cfg_path)
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
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

        with torch.no_grad():
            outputs = self.model(img)

        #print(outputs)
        boxes, confidences = outputs
        boxes = boxes.squeeze(0).squeeze(1)  
        confidences = confidences.squeeze(0).squeeze(1)

        conf_thresh = 0.9
        mask = confidences > conf_thresh
        boxes_high_conf = boxes[mask]
        #print(boxes_high_conf)

        for box in boxes_high_conf:
            box = box.tolist()
            x_norm, y_norm, w_norm, h_norm = box

                # Convert normalized [0,1] to pixels
            x = int(x_norm * 1280)
            y = int(y_norm * 720)
            w = int(w_norm * 1280)
            h = int(h_norm * 720)
            if x+30<1280 and y+30<720:
                zombie = [x+30,y+30,30,30]
            else:
                zombie = [x,y,30,30]
            matrix.append(zombie)

        return matrix

