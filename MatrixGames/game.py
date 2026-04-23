import numpy as np
import random
import math

class Game:
    def __init__(self,N, A, u, u_1, u_2): #similar to a game in normal form but here A is set of possible actions
        self._N = N
        self._A = A
        self._u = u
        self._u_1 = u_1
        self._u_2 = u_2
        self._alpha = 0.0001
        self._player_list = []

        self._trajectory_p1 = []
        self._trajectory_p2 = []

    @property 
    def N(self): 
	    return self._N
    
    @property 
    def A(self): 
        return self._A
    
    @property 
    def u(self): 
	    return self._u
    
    @property 
    def player_list(self): 
	    return self._player_list
    
    @property 
    def alpha(self): 
	    return self._alpha
    
    @property 
    def u_1(self): 
	    return self._u_1
    
    @property 
    def u_2(self): 
	    return self._u_2
    
    def get_last_trajectories(self):
         return self._trajectory_p1, self._trajectory_p2
    
    def empty_trajectories(self):
         self._trajectory_p1.clear()
         self._trajectory_p2.clear()

    def reinit_strategies(self):
         for player in self._player_list:
            #q1 = random.randint(1,10)
            #q2 = random.randint(1,10)
            q1 = random.random()
            q2 = random.random()
            player._q_table = [q1,q2]

    
    def add_player(self,player):
        if len(self._player_list) < len(self._N):
            self._player_list.append(player)
        else:
             raise ValueError("Cannot add more players.")
	
    def get_moves_utility(self, *args):
         print(len(args))
         if (len(args) != len(self._N)):
              raise ValueError("Not good amount of moves for this game.")
         return self._u[args]
    
    def play_round(self):
        if (len(self._player_list)==len(self._N)):
            #print("Making moves")
            moves = []
            moves_names = []
            for player in self._player_list:
                move = player.make_move()
                moves.append(move)
                moves_names.append(self._A[move])
            moves_tuple = tuple(moves)
            moves_names_tuple = tuple(moves_names)
            utility = self._u[moves_names_tuple]
            #print(utility)
            for i in range(len(self._player_list)):
                self._player_list[i].q_learning_update(moves_tuple[i],utility[i],self._alpha)
                if i==0:
                    self._trajectory_p1.append(self._player_list[i].strategy[0])
                elif i==1:
                    self._trajectory_p2.append(self._player_list[i].strategy[0])
            return utility
        
    
    
class Player:
    def __init__(self, game : Game,i):
        num_actions = len(game.A)
        p = 1.0/num_actions
        self._strategy = [p  for i in range(len(game.A))]
        self._q_table = [0.0 for i in range(len(self._strategy))]
        self._i = i
    
    @property 
    def strategy(self): 
	    return self._strategy
    
    @property 
    def q_table(self): 
	    return self._q_table
    
    @strategy.setter 
    def strategy(self, strategy): 
        #print("setter called")
	    self._strategy = strategy
    

    
    def make_move(self):
        number = np.random.uniform()
        cumulative = self._strategy[0]
        index = 0
        while (cumulative<number):
            cumulative+=self._strategy[index]
            index+=1
        if index > len(self._strategy)-1:
             index-=1
        return index
    
    def make_random_move(self):
        number = np.random.uniform()
        cumulative = 0
        index = 0
        while (cumulative<number):
            cumulative+=self._strategy[index]
            if number < cumulative:
                return index
            index+=1
        return len(self._strategy) - 1

    def q_learning_update(self, move, utility, alpha):
        self._q_table[move] = self._q_table[move]+alpha*(utility-self._q_table[move])
    


class EpsilonGreedyPlayer(Player):
     def __init__(self, game, i, epsilon):
          super().__init__(game,i)
          self._epsilon = epsilon

     def make_move(self):
        number = np.random.uniform()
        if number < self._epsilon:
            return super().make_random_move()
        else:
            max_q_value = max(self._q_table)
            indices = [i for i, val in enumerate(self._q_table) if val == max_q_value]
            return random.choice(indices)
    

class BoltzmannPlayer(Player):
     def __init__(self, game, i,temperature):
        super().__init__(game,i)
        self._temperature = temperature
        if len(super().strategy)==2:
             #q1 = random.randint(1,10)
             #q2 = random.randint(1,10)
             q1 = random.random()
             q2 = random.random()
             self._q_table = [q1,q2]
             

     @property 
     def temperature(self): 
	     return self._temperature

     def exp_list_temp(self,l):
        result = []
        for el in l:
            result.append(math.exp(el/self._temperature))
        return result
          
     def make_move(self):
        exp_q_values =self.exp_list_temp(self._q_table)
        sum_exps = sum(exp_q_values)
        strategy = [el/sum_exps for el in exp_q_values]
        
        self.strategy = strategy
        #print(self._strategy)

        move = super().make_random_move()
        #print(move)
        return move
        
class LenientBoltzmannPlayer(Player):
    def __init__(self, game, i, temperature, leniency): #leniency is N (updating q value based on maximum reward of N actions)
          super().__init__(game, i)
          self._temperature = temperature
          self._leniency = leniency
          '''self._last_N_rewards = []
          self._last_N_moves = []'''
          self._last_N_rewards = {}
          for action in range(len(game.A)):
               self._last_N_rewards[action] = []

          #q1 = random.randint(1,10)
          #q2 = random.randint(1,10)
          q1 = random.random()
          q2 = random.random()
          #q1 = 5
          #q2 = 5
          self._q_table = [q1,q2]

    @property 
    def temperature(self): 
	     return self._temperature
    
    def exp_list_temp(self,l):
        result = []
        for el in l:
            result.append(math.exp(el/self._temperature))
        return result
    
    def make_move(self):
        exp_q_values =self.exp_list_temp(self._q_table)
        sum_exps = sum(exp_q_values)
        strategy = [el/sum_exps for el in exp_q_values]
        
        self.strategy = strategy
        #print(self._strategy)

        move = super().make_random_move()
        #print(move)
        return move
    
    def add_reward(self,reward,move):
        if len(self._last_N_rewards[move]) < self._leniency:
            self._last_N_rewards[move].append(reward)

    def clear_rewards(self,move):
         self._last_N_rewards[move].clear()
    
    def q_learning_update(self, move, utility, alpha):
        self.add_reward(utility,move)
        if len(self._last_N_rewards[move])==self._leniency:
            max_reward = max(self._last_N_rewards[move])
            self._q_table[move] = self._q_table[move]+alpha*(max_reward-self._q_table[move])
            self.clear_rewards(move)


     
     
