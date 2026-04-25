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

def calculate_vector_field_lenient(game, P,Q, u1, u2, N):
    vector_field_x = np.zeros((100,100))
    vector_field_y = np.zeros((100,100))
    u2 = u2.T
    for a in range(100):
        for b in range(100):
            strategy1 = np.array([P[a,b], 1-P[a,b]])
            strategy2 = np.array([Q[a,b], 1-Q[a,b]])

            player1_expected_reward = np.zeros(shape = len(strategy1))
            player2_expected_reward = np.zeros(shape = len(strategy2))
            for i in range(len(game.A)):
                for j in range(len(game.A)):
                    p_j = strategy2[j]
                    a_ij = u1[i,j]
                    sum1 = sum(
                        strategy2[k]
                        for k in range(len(strategy2))
                        if u1[i,k] == a_ij
                    )
                    sum2 = sum(
                        strategy2[k]
                        for k in range(len(strategy2))
                        if u1[i,k] <= a_ij
                    )
                    exp_sum2 = sum2 ** N
                    sum3 = sum(
                        strategy2[k]
                        for k in range(len(strategy2))
                        if u1[i,k] < a_ij
                    )
                    exp_sum3 = sum3 ** N
                    player1_expected_reward[i] += (a_ij*p_j/sum1)*(exp_sum2-exp_sum3)
            for i in range(len(game.A)):
                for j in range(len(game.A)):
                    p_j = strategy1[j]
                    a_ij = u2[i,j]
                    sum1 = sum(
                        strategy1[k]
                        for k in range(len(strategy1))
                        if u2[i,k] == a_ij
                    )
                    sum2 = sum(
                        strategy1[k]
                        for k in range(len(strategy1))
                        if u2[i,k] <= a_ij
                    )
                    exp_sum2 = sum2 ** N
                    sum3 = sum(
                        strategy1[k]
                        for k in range(len(strategy1))
                        if u2[i,k] < a_ij
                    )
                    exp_sum3 = sum3 ** N
                    player2_expected_reward[i] += (a_ij*p_j/sum1)*(exp_sum2-exp_sum3)

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
            vector_field_x[a,b] = derivative_player1
            vector_field_y[a,b] = derivative_player2
    return vector_field_x,vector_field_y
def calculate_vector_field_boltzmann_one_pop(game, P,Q, u1, u2):
    vector_field_x = np.zeros((100,100))
    vector_field_y = np.zeros((100,100))
    u2 = u2.T
    V1 = np.array([0.0, 0.0])
    V2 = np.array([1.0, 0.0])
    V3 = np.array([0.5, np.sqrt(3)/2])

    for a in range(100):
        for b in range(100):
            # map grid → [0,1]
            x = a / 99
            y = b / 99
            p = np.array([x, y])

            T = np.column_stack((V2 - V1, V3 - V1))
            v = p - V1

            try:
                l2, l3 = np.linalg.solve(T, v)
            except:
                vector_field_x[a, b] = np.nan
                vector_field_y[a, b] = np.nan
                continue

            l1 = 1 - l2 - l3

            if (l1 < 0) or (l2 < 0) or (l3 < 0):
                vector_field_x[a, b] = np.nan
                vector_field_y[a, b] = np.nan
                continue

            strategy1 = np.array([l1,l2,l3])
            player1_expected_reward = u1 @ strategy1
            

            helper_term1 = strategy1/strategy1[0]
            helper_term1 = np.log(helper_term1)

            helper_term2 = strategy1/strategy1[1]
            helper_term2 = np.log(helper_term2)

            helper_term3 = strategy1/strategy1[2]
            helper_term3 = np.log(helper_term3)

            d1 =(
                (game.alpha*strategy1[0])/game.player_list[0].temperature
                *(player1_expected_reward[0]-player1_expected_reward @ strategy1)
                +game.alpha*strategy1[0]*helper_term1 @ strategy1
                )
            
            d2 =(
                (game.alpha*strategy1[1])/game.player_list[0].temperature
                *(player1_expected_reward[1]-player1_expected_reward @ strategy1)
                +game.alpha*strategy1[1]*helper_term2 @ strategy1
                )
            
            d3 =(
                (game.alpha*strategy1[2])/game.player_list[0].temperature
                *(player1_expected_reward[2]-player1_expected_reward @ strategy1)
                +game.alpha*strategy1[2]*helper_term3 @ strategy1
                )
            
           
            # optional: enforce simplex
            d_sum = (d1 + d2 + d3) / 3
            d1 -= d_sum
            d2 -= d_sum
            d3 -= d_sum

            # barycentric → cartesian
            v_cart = d1*V1 + d2*V2 + d3*V3

            vector_field_x[a,b] = v_cart[0]
            vector_field_y[a,b] = v_cart[1]

    return vector_field_x, vector_field_y
            

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
    
    
def show_replicator_dynamics_boltzmann_one_pop(game : game.Game):
    p = np.linspace(0.01,1,100)
    q = np.linspace(0.01,1,100)
    P, Q = np.meshgrid(p, q)

    utility_mat_1 = game.u_1
    utility_mat_2 = game.u_2
    
    print(P)
    print(Q)
    
    v1, v2 = calculate_vector_field_boltzmann_one_pop(game,P,Q,utility_mat_1,utility_mat_2)
    fig, ax = plt.subplots()
    ax.quiver(P[::5,::5], Q[::5,::5], v1[::5,::5], v2[::5,::5],units='xy')
    ax.set_aspect('equal')
    return ax
    # show plot
    #plt.show()
    


def show_replicator_dynamics_lenient_boltzmann(game : game.Game):
    p = np.linspace(0.01,1-0.01,100)
    q = np.linspace(0.01,1-0.01,100)
    P, Q = np.meshgrid(p, q)

    utility_mat_1 = game.u_1
    utility_mat_2 = game.u_2
    
    print(P)
    print(Q)
    
    v1, v2 = calculate_vector_field_lenient(game,P,Q,utility_mat_1,utility_mat_2,10)
    fig, ax = plt.subplots()
    ax.quiver(P[::5,::5], Q[::5,::5], v1[::5,::5], v2[::5,::5],units='xy')
    ax.set_aspect('equal')
    return ax