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
from NeuralNet.AngleNet import AngleNet
from gymnasium import spaces

class CustomWrapper(BaseWrapper):
    """
    Wrapper to use to add state pre-processing (feature engineering)
    """
    def __init__(self, env):
        super().__init__(env)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AngleNet().to(self.device)
        self.model.load_state_dict(torch.load("./NeuralNet/anglenet.pth", map_location=self.device))
        self.model.eval()
        self.zombie_detector = CustomZombieDetectorFunction(self.env)

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

        x_min = x - 20
        x_max = x + 21
        y_min = y - 20
        y_max = y + 21
                 
        x_min = max(0, x_min)
        x_max = min(1280, x_max)
        y_min = max(0, y_min)
        y_max = min(720, y_max)

        crop = obs[y_min:y_max, x_min:x_max, :]
        h, w, _ = crop.shape

        pad_h = max(0, 20 - h)
        pad_w = max(0, 20 - w)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        crop = np.pad(
            crop,
            ((max(0, 41-h)//2, max(0, 41-h) - max(0, 41-h)//2),
            (max(0, 41-w)//2, max(0, 41-w) - max(0, 41-w)//2),
            (0,0)),
            mode='constant',
            constant_values=0
        )
        
        
        self.model.eval()
        crop_tensor = ((torch.from_numpy(crop).float()/255.0)-0.5)*2
        crop_tensor = crop_tensor.permute(2, 0, 1)
        crop_tensor = crop_tensor.unsqueeze(0) 
        
        with torch.no_grad():
            output_infer = self.model(crop_tensor)
            output_infer = output_infer.squeeze(0)
        zombies = self.zombie_detector(obs)
        #zombies = []
        num_zombies = len(zombies)
        final_obs = [x*2/1280-1,y*2/720-1,
                         output_infer[0].item(),output_infer[1].item(),
                         teammate_pos[0]*2/1280-1,teammate_pos[1]*2/720-1]
        if num_zombies<5:
            for i in range(num_zombies):
               final_obs.extend([zombies[i][0]*2/1280-1,zombies[i][1]*2/720-1,1.0])
            for i in range(5-num_zombies):
                final_obs.extend([0.0,0.0,-1.0])
        if num_zombies >= 5:
            for i in range(5):
               final_obs.extend([zombies[i][0]*2/1280-1,zombies[i][1]*2/720-1,1.0])
        #print(final_obs)
        return np.array(final_obs, dtype=np.float32)
            


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
        self.net = None
        

    def __call__(self, observation, *args, **kwargs):
        """Returns a matrix of shape (nb_zombies, nb_attributes), where
        the attributes are defining a rectangle with (x,y,width,heigh) and
        indicate where the zombies are. The zombies are ordered from most
        likely to least likely positions. The evaluation uses the first k
        items if there are k zombies on the screen.
        """
        if self.net is None:
            self.net = cv2.dnn.readNet(
                "yolov4-tiny-weights/yolov4-tiny-obj_best(1).weights",
                "yolov4-tiny-obj.cfg"
            )
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        matrix = []
        
        blob = cv2.dnn.blobFromImage(observation, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        conf_threshold = 0.5
        nms_threshold = 0.4
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
                        
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                        
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        
        for i in indices:
            idx = i[0] if isinstance(i, (list, np.ndarray)) else i
            
            x, y, w, h = boxes[idx]
            conf = confidences[idx]
            
            matrix.append([x,y,w,h])
        return matrix

