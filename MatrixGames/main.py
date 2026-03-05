import game

def main():
    print("Hello World!")
    #Game 1 - Stag Hunt
    N = {1,2}
    A = ['S','H']
    u = {('S','S'):[1, 1],
         ('S','H'):[0, 2.0/3],
         ('H','S'):[2.0/3, 0],
         ('H','H'):[2.0/3, 2.0/3]}
    
    stag_hunt = game.Game(N,A,u)
    '''player1 = game.EpsilonGreedyPlayer(stag_hunt,1,0.1)
    player2 = game.Player(stag_hunt,2)
    player2._strategy = [0.8,0.2]
    stag_hunt.add_player(player1)
    stag_hunt.add_player(player2)'''

    player1 = game.BoltzmannPlayer(stag_hunt,1,3)
    player2 = game.Player(stag_hunt,2)
    player2._strategy = [0.8,0.2]
    stag_hunt.add_player(player1)
    stag_hunt.add_player(player2)
    
    for i in range(10000):
        stag_hunt.play_round()
        if i%1000==0:
            print(player1.q_table)
            print(player2.q_table)



if __name__ == "__main__":
    main()