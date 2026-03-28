"""Template of your submission file for Task 3 (multi agent KAZ).
"""
from typing import Callable
import gymnasium
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType
from PIL import Image
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
        self.net = cv2.dnn.readNet("yolov4-tiny-weights/yolov4-tiny-obj_best(1).weights", "yolov4-tiny-obj.cfg")
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def __call__(self, observation, *args, **kwargs):
        """Returns a matrix of shape (nb_zombies, nb_attributes), where
        the attributes are defining a rectangle with (x,y,width,heigh) and
        indicate where the zombies are. The zombies are ordered from most
        likely to least likely positions. The evaluation uses the first k
        items if there are k zombies on the screen.
        """
        matrix = []
        
        blob = cv2.dnn.blobFromImage(observation, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        conf_threshold = 0.5
        nms_threshold = 0.4 # Non-Maximum Suppression to remove double-boxes
        boxes = []
        confidences = []

        for out in outputs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                    
                if confidence > conf_threshold:
                    center_x = int(detection[0] * 1280)+15
                    center_y = int(detection[1] * 720)+15
                    w = int(detection[2] * 1280)
                    h = int(detection[3] * 720)
                        
                        # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                        
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        
        for i in indices:
            # This line handles both old (nested) and new (flat) OpenCV versions
            idx = i[0] if isinstance(i, (list, np.ndarray)) else i
            
            # Now use that index to grab the original data
            x, y, w, h = boxes[idx]
            conf = confidences[idx]
            
            matrix.append([x,y,w,h])
        return matrix

