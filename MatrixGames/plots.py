import numpy as np
import matplotlib.pyplot as plt
import game
import math

def create_simple_plot():
    # Creating arrow
    x_pos = 0
    y_pos = 0
    x_direct = 1
    y_direct = 1

    # Creating plot
    fig, ax = plt.subplots(figsize = (12, 7))
    ax.quiver(x_pos, y_pos, x_direct, y_direct)
    ax.set_title('Quiver plot with one arrow')

    # Show plot
    plt.show()

def calculate_vector_field_boltzmann(game, P,Q, u1, u2):
    vector_field_x = np.zeros((100,100))
    vector_field_y = np.zeros((100,100))

    for i in range(100):
        for j in range(100):
            strategy1 = np.array([P[i,j], 1-P[i,j]])
            strategy2 = np.array([Q[i,j], 1-Q[i,j]])

            player1_expected_reward = u1 @ strategy2
            player2_expected_reward = u2.T @ strategy1

            helper_term = strategy1/strategy1[0]
            helper_term = np.log(helper_term)
            derivative_player1 =(
                (game.alpha*strategy1[0])/game.player_list[0].temperature
                *(player1_expected_reward[0]-player1_expected_reward @ strategy1)
                +game.alpha*strategy1[0]*helper_term @ strategy1
                )
            
            helper_term = strategy2/strategy2[0]
            helper_term = np.log(helper_term)
            derivative_player2 = (
                (game.alpha*strategy2[0])/game.player_list[1].temperature
                *(player2_expected_reward[0]-player2_expected_reward @ strategy2)
                +game.alpha*strategy2[0]*helper_term @ strategy2
                )
            vector_field_x[i,j] = derivative_player1
            vector_field_y[i,j] = derivative_player2
    return vector_field_x,vector_field_y




def show_replicator_dynamics_boltzmann(game : game.Game):
    p = np.linspace(0.01,1,100)
    q = np.linspace(0.01,1,100)
    P, Q = np.meshgrid(p, q)

    utility_mat_1 = game.u_1
    utility_mat_2 = game.u_2
    
    print(P)
    print(Q)
    
    v1, v2 = calculate_vector_field_boltzmann(game,P,Q,utility_mat_1,utility_mat_2)
    fig, ax = plt.subplots()
    ax.quiver(P[::5,::5], Q[::5,::5], v1[::5,::5], v2[::5,::5],units='xy')
    ax.set_aspect('equal')
    return ax
    # show plot
    #plt.show()
    
    

    


def show_replicator_dynamics_lenient_boltzmann(game : game.Game):
    pass