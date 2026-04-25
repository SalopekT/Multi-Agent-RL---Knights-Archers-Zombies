import game
import plots
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("Hello World!")
    #Game 1 - Stag Hunt
    N = {1,2}
    A = ['S','H']
    u = {('S','S'):[1, 1],
         ('S','H'):[0, 2.0/3],
         ('H','S'):[2.0/3, 0],
         ('H','H'):[2.0/3, 2.0/3]}
    u_1 = np.array([[1,0],[2.0/3,2.0/3]])
    u_2 = np.array([[1,2.0/3],[0,2.0/3]])
    stag_hunt = game.Game(N,A,u,u_1,u_2)

    #Game 2 - Subsidy game
    N_s = {1,2}
    A_s = ['S1','S2']
    u_s = {('S1','S1'):[12, 12],
         ('S1','S2'):[0, 11],
         ('S2','S1'):[11, 0],
         ('S2','S2'):[10, 10]}
    u_1_s = np.array([[12,0],[11,10]])
    u_2_s = np.array([[12,11],[0,10]])
    subsidy = game.Game(N_s,A_s,u_s,u_1_s,u_2_s)

    #Game 3 - Prisoner's dillema
    N_p = {1,2}
    A_p = ['C','D']
    u_p = {('C','C'):[-1, -1],
         ('C','D'):[-4,0],
         ('D','C'):[0, -4],
         ('D','D'):[-3, -3]}
    u_1_p = np.array([[-1,-4],[0,-3]])
    u_2_p = np.array([[-1,0],[-4,-3]])
    prisoners = game.Game(N_p,A_p,u_p,u_1_p,u_2_p)

    #Game 4 - Rock Paper Scissors
    N_rps = {1,2}
    A_rps = ['R','P','S']
    u_rps = {('R','R'):[0, 0],
         ('R','P'):[-0.05,-0.05],
         ('R','S'):[0.25, 0.25],
         ('P','R'):[0.05, 0.05],
         ('P','P'):[0, 0],
         ('P','S'):[-0.5, -0.5],
         ('S','R'):[-0.25, -0.25],
         ('S','P'):[0.5, 0.5],
         ('S','S'):[0, 0]}
    u_1_rps = np.array([[0,-0.05,0.25],[0.05,0,-0.5],[-0.25,0.5,0]])
    u_2_rps = np.array([[0,-0.05, 0.25],[0.05, 0,-0.5],[-0.25,0.5,0]])
    
    rps = game.Game(N_rps,A_rps,u_rps,u_1_rps,u_2_rps)
    '''player1 = game.EpsilonGreedyPlayer(stag_hunt,1,0.1)
    player2 = game.EpsilonGreedyPlayer(stag_hunt,2,0.1)
    
    stag_hunt.add_player(player1)
    stag_hunt.add_player(player2)'''

    player1 = game.BoltzmannPlayer(rps,1,0.1)
    #player1._q_table = [10,1.9/3]

    player2 = game.BoltzmannPlayer(rps,2,0.1)
    #player2._q_table = [10,1.9/3]

    '''player3 = game.Player(stag_hunt,2)
    player3.strategy = [0.7,0.3]'''

    '''player1 = game.LenientBoltzmannPlayer(prisoners,1,0.1,10)
    player2 = game.LenientBoltzmannPlayer(prisoners,2,0.1,10)'''

    rps.add_player(player1)
    #rps.add_player(player1)


    ax = plots.show_replicator_dynamics_boltzmann_one_pop(rps)
    
    for epoch in range(5):
        for i in range(1000000):
            rps.play_round()
            if i%100000==0:
                print(player1.q_table)
                print(player2.q_table)
        trajectory_p1, trajectory_p2 = rps.get_last_trajectories()
        ax.plot(trajectory_p1,trajectory_p2,'r')
        rps.empty_trajectories()
        rps.reinit_strategies()
    
    plt.show()


    '''ax = plots.show_replicator_dynamics_boltzmann(stag_hunt)
    
    for epoch in range(5):
        for i in range(1000000):
            rps.play_round()
            if i%100000==0:
                print(player1.q_table)
                print(player2.q_table)
        trajectory_p1, trajectory_p2 = rps.get_last_trajectories()
        ax.plot(trajectory_p1,trajectory_p2,'r')
        rps.empty_trajectories()
        rps.reinit_strategies()

    plt.show()'''
    '''ax = plots.show_replicator_dynamics_lenient_boltzmann(prisoners)
    
    for epoch in range(5):
        for i in range(1000000):
            prisoners.play_round()
            if i%100000==0:
                print(player1.q_table)
                print(player2.q_table)
        trajectory_p1, trajectory_p2 = prisoners.get_last_trajectories()
        ax.plot(trajectory_p1,trajectory_p2,'r')
        prisoners.empty_trajectories()
        prisoners.reinit_strategies()

    plt.show()'''
    
    '''for i in range(1000000):
        stag_hunt.play_round()
        if i%100000==0:
            print(player1.q_table)
            print(player2.q_table)'''

if __name__ == "__main__":
    main()