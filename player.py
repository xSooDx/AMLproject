
import gym
import numpy as np
from model import InceptionDQAgent


EPISODES = 100

if __name__ == "__main__":
    env = gym.make("SpaceInvaders-v0")
    action_size = env.action_space.n
    print(action_size)
    #set explore_factor to 1 for random agent
    agent = InceptionDQAgent(action_size, explore_factor=0.0)
    agent.load("spaceinvaders-inception.h5")
    print("loaded")
    EPISODES = 200
    
    for e in range(EPISODES):
        print("Ep:", e)
        time=0
        score = 0
        lives = 3
        done = False
        state = env.reset()
        while not done:
            t_state = np.reshape(np.pad(state,((3,3),(0,0),(0,0)), 'constant', constant_values=0),[1,216,160,3])
            action = agent.act(t_state)
            next_state, reward, done, info = env.step(action)
            if not reward == 0:
                print("Reward:",reward)
            state = next_state
            env.render()