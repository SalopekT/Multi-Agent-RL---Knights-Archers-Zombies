from typing import Callable
import random
import numpy as np
import gymnasium
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType
from PIL import Image
import pygame

class CustomWrapper(BaseWrapper):
    """
    An example of a custom wrapper that flattens the symbolic vector state of the environment.

    Wrappers are useful to do state pre-processing (e.g. feature engineering) that does not need to be learned by the agent.
    """
    def __init__(self, env):
        super().__init__(env)

        self.angle_archer0 = 0
        self.angle_archer1 = 0
        self.archer0_direction = pygame.Vector2(0, -1)
        self.archer1_direction = pygame.Vector2(0, -1)
        self.ang_rate = 10
        self.phase_0 = 0
        self.phase_1 = 0

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return spaces.Box(
        low=np.array([0, 0]),
        high=np.array([360, 10]),
        dtype=np.int32
    )

    def observe(self, agent: AgentID) -> ObsType | None:
        if agent == "archer_0":
            angle = int(self.angle_archer0)
            phase = int(self.phase_0)
        else:
            angle = int(self.angle_archer1)
            phase = int(self.phase_1)


        return np.array([angle, phase], dtype=np.int32)


        '''obs = super().observe(agent)
        img = Image.fromarray(obs, mode='RGB') 
        img.save('rgb.png') 
        #print(obs.shape)
        #flat_obs = obs.flatten()
        #print(flat_obs.shape)
        return obs'''

    def step(self, action):
        agent = self.env.agent_selection
        archers = list(self.env.unwrapped.archer_list)
        if len(archers)==2:
            if (agent == "archer_0"):
                own_pos = archers[0].rect.center
                teammate_pos = archers[1].rect.center
            else:
                own_pos = archers[1].rect.center
                teammate_pos = archers[0].rect.center
               
        
        x, y = own_pos
        if action==2: 
            if agent == "archer_0":
                
                self.angle_archer0 += self.ang_rate
                self.archer0_direction = pygame.Vector2(0, -1).rotate(-self.angle_archer0)
                if self.phase_0 == 0 and abs(self.angle_archer0)>=90:
                    self.phase_0=1
                if self.phase_0 >= 2 and self.phase_0 <= 18:
                    self.phase_0+=1
            else:
                
                self.angle_archer1 += self.ang_rate
                self.archer1_direction = pygame.Vector2(0, -1).rotate(-self.angle_archer1)
                if self.phase_1 == 0 and abs(self.angle_archer1)>=90:
                    self.phase_1=1
                if self.phase_1 >= 2 and self.phase_1 <= 18:
                    self.phase_1+=1
        elif action==3:
            if agent == "archer_0":
                
                self.angle_archer0 -= self.ang_rate
                self.archer0_direction = pygame.Vector2(0, -1).rotate(-self.angle_archer0)
                if self.phase_0 == 0 and abs(self.angle_archer0)>=90:
                    self.phase_0=1
                if self.phase_0 >= 2 and self.phase_0 <= 18:
                    self.phase_0+=1
            else:
                
                self.angle_archer1 -= self.ang_rate
                self.archer1_direction = pygame.Vector2(0, -1).rotate(-self.angle_archer1)
                if self.phase_1 == 0 and abs(self.angle_archer1)>=90:
                    self.phase_1=1
                if self.phase_1 >= 2 and self.phase_1 <= 18:
                    self.phase_1+=1
        
        elif action == 0:
            print(x)
            if agent == "archer_0":
                if self.phase_0 == 1 and x<=120:
                    self.phase_0 = 2
            else:
                if self.phase_1 == 1 and x>=1140:
                    self.phase_1 = 2
        #print(self.angle_archer0)
        print(action)
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
    """A random archer agent."""

    def __init__(self, env):
        self.env = env

    def __call__(self, observation, agent, *args, **kwargs):
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
        if agent=="archer_0":
            if observation[1]==0:
                action = 2
            if observation[1]==1:
                action = 0
            if observation[1]>=2 and observation[1]<=18:
                action = 3
            if observation[1]>18:
                action = 4
        if agent=="archer_1":
            if observation[1]==0:
                action = 3
            if observation[1]==1:
                action = 0
            if observation[1]>=2 and observation[1]<=18:
                action = 2
            if observation[1]>18:
                action = 4


        return action

class CustomZombieDetectorFunction(Callable):
    """Returns random detections."""

    def __init__(self, env: gymnasium.Env):
        pass

    def __call__(self, observation, *args, **kwargs):
        print("hello")
        print(observation)
        nb_zombies_detected = random.randint(0,4)
        zombie_rects = np.zeros((nb_zombies_detected, 4))
        for i in range(nb_zombies_detected):
            x = random.randint(0,1280-29)
            y = random.randint(0,720-31)
            w, h = 29, 31
            zombie_rects[i, :] = [x, y, w, h]
        return zombie_rects
