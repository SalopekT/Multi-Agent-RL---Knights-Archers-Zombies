import numpy as np
import matplotlib.pyplot as plt
import game


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
    vector_field_x = np.zeros((11,11))
    vector_field_y = np.zeros((11,11))

    for i in range(11):
        for j in range(11):
            strategy1 = np.array([P[i,j], 1-P[i,j]])
            strategy2 = np.array([Q[i,j], 1-Q[i,j]])

            player1_expected_reward = u1 @ strategy2
            player2_expected_reward = u2 @ strategy1

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
    p = np.arange(0, 1.1, 0.1)
    q = np.arange(0, 1.1, 0.1)  
    P, Q = np.meshgrid(p, q)

    utility_mat_1 = game.u_1
    utility_mat_2 = game.u_2
    
    print(P)
    print(Q)
    
    v1, v2 = calculate_vector_field_boltzmann(game,P,Q,utility_mat_1,utility_mat_2)
    fig, ax = plt.subplots(figsize =(14, 8))
    ax.quiver(P, Q, v1, v2)

    '''ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])'''
    #ax.axis([-0.3, 2.3, -0.3, 2.3])
    ax.set_aspect('equal')

    # show plot
    plt.show()
    
    

    


def show_replicator_dynamics_lenient_boltzmann(game : game.Game):
    pass