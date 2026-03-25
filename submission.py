"""Template of your submission file for Task 3 (multi agent KAZ).
"""
from typing import Callable
import gymnasium
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType
from PIL import Image
from ultralytics import YOLO
import template_matching as tm
from pathlib import Path
from ray.rllib.core.rl_module import MultiRLModule
import torch
import numpy as np
import cv2

class CustomWrapper(BaseWrapper):
    """
    Wrapper to use to add state pre-processing (feature engineering)
    """

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return spaces.flatten_space(super().observation_space(agent))

        '''max_zombies = self.env.unwrapped.max_zombies
        box = spaces.Box(low=-1, high=512, shape=(max_zombies,2), dtype=np.int8)
        return box'''

    def observe(self, agent: AgentID) -> ObsType | None:
        max_zombies = self.env.unwrapped.max_zombies
        #obs = self.env.unwrapped.observe(agent)
        obs = self.env.observe(agent)
        state = self.env.state()

        archers = list(self.env.unwrapped.archer_list)
        if (agent == "archer_0"):
            own_pos = archers[0].rect.center
            teammate_pos = archers[1].rect.center
        else:
            own_pos = archers[1].rect.center
            teammate_pos = archers[0].rect.center
        x, y = own_pos

        crop = obs[y-15:y+15, x-15:x+15]
        
        '''padded_image = cv2.copyMakeBorder(
                    state,
                    top=256, bottom=256,
                    left=256, right=256,
                    borderType=cv2.BORDER_CONSTANT,
                    value=[0,0,0]
        )
        
        minLoc = tm.observations_matching(obs,padded_image)
        if agent == "archer_0":
            cv2.circle(padded_image, (minLoc[0],minLoc[1]), 5, (0,0,255), -1)
            detector = CustomZombieDetectorFunction(self.env)
            boxes = detector(state)
            for box in boxes:
                print(box)
                x, y, w, h = box
                zx = x + w // 2 + 256
                zy = y + h // 2 + 256
                cv2.circle(padded_image, (int(zx), int(zy)), 5, (255, 0, 0), -1)
            
            cv2.imwrite("obs_image.png", obs)
            cv2.imwrite("state_image.png", padded_image)

            x, y = minLoc
            x+=20
            y+=20
            crop_size = 36
            half = crop_size // 2
            x1 = max(x - half, 0)
            y1 = max(y - half, 0)
            x2 = x + half
            y2 = y + half
            player_crop = padded_image[y1:y2, x1:x2].copy()
            cv2.imwrite("player_crop.png", player_crop)

        return state'''


class CustomPredictFunction(Callable):
    """Function to use to load the trained model and predict the action"""

    '''def __init__(self, env):

        # Here you should load your trained model(s) from a checkpoint in your folder
        best_checkpoint = (Path("results") / "learner_group" / "learner" / "rl_module").resolve()
        self.modules = MultiRLModule.from_checkpoint(best_checkpoint)

    def __call__(self, observation, agent, *args, **kwargs):
        rl_module = self.modules[agent]
        fwd_ins = {"obs": torch.Tensor(observation).unsqueeze(0)}
        fwd_outputs = rl_module.forward_inference(fwd_ins)
        action_dist_class = rl_module.get_inference_action_dist_cls()
        action_dist = action_dist_class.from_logits(
            fwd_outputs["action_dist_inputs"]
        )
        action = action_dist.sample()[0].numpy()
        return action'''
    
    def __init__(self, env):
        self.env = env

    def __call__(self, observation, agent, *args, **kwargs):
        return self.env.action_space(agent).sample()


class CustomZombieDetectorFunction(Callable):
    """Function to use to load the trained model and predict where
    the zombies are.
    """

    def __init__(self, env: gymnasium.Env):
        self._model =  YOLO("weights_vision3/best (5).pt")

    def __call__(self, observation, *args, **kwargs):
        """Returns a matrix of shape (nb_zombies, nb_attributes), where
        the attributes are defining a rectangle with (x,y,width,heigh) and
        indicate where the zombies are. The zombies are ordered from most
        likely to least likely positions. The evaluation uses the first k
        items if there are k zombies on the screen.
        """
        '''print(observation)
        results = self._model.predict(observation,verbose=False)
        matrix = []
        b_boxes = results[0].boxes.xywh
        for b_box in b_boxes:
            real_center_x = b_box[0]+15
            real_center_y = b_box[1]+15
            matrix.append(real_center_x,real_center_y,30,30)'''
        
        img = observation.astype(np.uint8)

        # Convert BGR → RGB if using OpenCV
        if img.shape[2] == 3:  # 3 channels
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        matrix = []
        #boxes, indices = tm.find_zombies(observation)
        results = self._model.predict(img, imgsz=416) 
        boxes = results[0].boxes
        for box in boxes:
            matrix.append(box.xywh[0])
        
        return matrix

