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


class CustomWrapper(BaseWrapper):

    def __init__(self, env, target_size=(64, 64)):
        super().__init__(env)
        self.target_size = target_size  # (H, W)

    def observation_space(self, agent: AgentID):
        return spaces.Box(low = -1.0, high = 1.0, shape = (3*2+5*3,))

    def observe(self, agent: AgentID) -> ObsType | None:
        obs = super().observe(agent)
        state = super().state()
        print(state)
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
        zombie_y_values = []
        for i in range(num_zombies):
            zombie_rel_x = zombies[i][0] - x
            zombie_rel_y = zombies[i][1] - y
            zombie_y_values.append(zombies[i][0])
        sorted_zombies = sorted(zombies, key=lambda z: z[1])
        #print("----------")
        sorted_zombies.reverse()

        final_obs = [x/1280,y/720,
                         output_heading[0],output_heading[1],
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
        print(final_obs)
        return np.array(final_obs, dtype=np.float32)


class CustomPredictFunction(Callable):
    """ This is an example of an instantiation of the CustomPredictFunction that loads a trained RLLib algorithm from
    a checkpoint and extract the policies from it"""

    def __init__(self, env):

        # Here you should load your trained model(s) from a checkpoint in your folder
        best_checkpoint = (Path("results") / "learner_group" / "learner" / "rl_module").resolve()
        self.modules = MultiRLModule.from_checkpoint(best_checkpoint)

    def __call__(self, observation, agent, *args, **kwargs):
        rl_module = self.modules["shared_policy"]
        fwd_ins = {"obs": torch.Tensor(observation).unsqueeze(0)}
        fwd_outputs = rl_module.forward_inference(fwd_ins)
        action_dist_class = rl_module.get_inference_action_dist_cls()
        action_dist = action_dist_class.from_logits(
            fwd_outputs["action_dist_inputs"]
        )
        action = action_dist.sample()[0].numpy()
        return action


class CustomZombieDetectorFunction(Callable):
    """Returns random detections."""

    def __init__(self, env: gymnasium.Env):
        pass

    def __call__(self, observation, *args, **kwargs):
        nb_zombies_detected = random.randint(0,4)
        zombie_rects = np.zeros((nb_zombies_detected, 4))
        for i in range(nb_zombies_detected):
            x = random.randint(0,1280-29)
            y = random.randint(0,720-31)
            w, h = 29, 31
            zombie_rects[i, :] = [x, y, w, h]
        return zombie_rects

