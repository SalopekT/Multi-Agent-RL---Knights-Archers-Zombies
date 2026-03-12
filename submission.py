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

class CustomWrapper(BaseWrapper):
    """
    Wrapper to use to add state pre-processing (feature engineering)
    """

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return spaces.flatten_space(super().observation_space(agent))

    def observe(self, agent: AgentID) -> ObsType | None:
        obs = super().observe(agent)

        
        return obs


class CustomPredictFunction(Callable):
    """Function to use to load the trained model and predict the action"""

    def __init__(self, env):

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
        return action


class CustomZombieDetectorFunction(Callable):
    """Function to use to load the trained model and predict where
    the zombies are.
    """

    def __init__(self, env: gymnasium.Env):
        #self._model =  YOLO("weights_vision/best(3).pt")
        pass

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
        
        
        matrix = []
        boxes, indices = tm.find_zombies(observation)
        
        for box in boxes:
            matrix.append(box)
        return matrix

