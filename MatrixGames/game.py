import numpy as np
import random
class Game:
    def __init__(self,N, A, u): #similar to a game in normal form but here A is set of possible actions
        self._N = N
        self._A = A
        self._u = u
        self._player_list = []

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
                if player.is_q_learning:
                    move = player.make_epsilon_greedy_move(0.2)
                else:
                     move = player.make_random_move()
                moves.append(move)
                moves_names.append(self._A[move])
            moves_tuple = tuple(moves)
            moves_names_tuple = tuple(moves_names)
            utility = self._u[moves_names_tuple]
            for i in range(len(self._player_list)):
                if self._player_list[i].is_q_learning:
                    self._player_list[i].q_learning_update(moves_tuple[i],utility[i],0.01)
            return utility
        
         
    
class Player:
    def __init__(self, game : Game, i, is_q_learning : bool):
        num_actions = len(game.A)
        p = 1.0/num_actions
        self._strategy = [p  for i in range(len(game.A))]
        self._q_table = [0.0 for i in range(len(self._strategy))]
        self._is_q_learning = is_q_learning
    
    @property 
    def strategy(self): 
	    return self._strategy
    
    @property 
    def q_table(self): 
	    return self._q_table
    
    @property 
    def is_q_learning(self): 
	    return self._is_q_learning
    
    def make_random_move(self):
        number = np.random.uniform()
        cumulative = self._strategy[0]
        index = 0
        while (cumulative<number):
            cumulative+=self._strategy[index]
            index+=1
        return index
    
    def make_epsilon_greedy_move(self,epsilon):
        number = np.random.uniform()
        if number < epsilon:
            return self.make_random_move() 
        else:
            max_q_value = max(self._q_table)
            indices = [i for i, val in enumerate(self._q_table) if val == max_q_value]
            return random.choice(indices)

    def q_learning_update(self, move, utility, alpha):
        self._q_table[move] = self._q_table[move]+alpha*(utility-self._q_table[move])
    
