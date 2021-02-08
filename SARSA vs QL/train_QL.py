import numpy as np
import matplotlib.pyplot as plt
import gym
import pickle as pk
from collections import defaultdict 
np.random.seed(2021)
import warnings
warnings.filterwarnings("ignore")

def train_QL(self, num_games = 100001, alpha = 0.1, \
    gamma = 0.9, epsilon = 1.0, temporary_game_check = 20000, plot_result = True):
    '''
    alpha: learning rate 
    gamma: disccounted rate
    epsilon: exploration ratio
    temporary_game_check: print out results every "temporary_game_check" games
    num_games: number of trials
    '''

    # screening & initializing
    method = self.method # QL
    successful_steps = []

    for i in range(num_games):
        # printing to check process
        if i % temporary_game_check == 0:
            print('starting game', i)

        # foreach game, reset the environment      
        # cart x position, cart velocity, pole theta, pole velocity
        observation = self.env.reset()   
            
        s = self.get_discrete(observation)

        # e-greedy policy: epsilon chance of exploring and (1-epsilon) chance of exploiting 
        rand = np.random.random()
        a = self.max_action(s) if rand < (1 - epsilon) else self.env.action_space.sample()
        done = False

        # keep playing
        while not done:
            '''
            Move to the next state, use the next state value to update back the current state
            '''
            
            # Move to next state
            observation_, reward, done, info = self.env.step(a)   
            s_ = self.get_discrete(observation_)

            # update (depends on learning method)
            # off policy
            self.Q_table[method][s,a] = self.Q_table[method][s,a] + alpha*(reward + gamma*self.Q_table[method][s_, self.max_action(s_)] - self.Q_table[method][s,a])

            rand = np.random.random()
            a_ = self.max_action(s_) if rand < (1-epsilon) else self.env.action_space.sample()

            # update next state
            s, a = s_, a_ 

        # epsilon decay every time so we explore less later in the game
        epsilon *= (num_games - 1)/num_games
    return None
    