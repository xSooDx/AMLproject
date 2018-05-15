from model import InceptionDQAgent
import gym
import numpy as np

if __name__ == "__main__":
    env = gym.make("SpaceInvaders-v0")
    action_size = env.action_space.n
    print(action_size)
    agent = InceptionDQAgent(action_size)
    
    agent.load("spaceinvaders-inception.h5")
    
    EPISODES = 1000
    
    for e in range(EPISODES):
        print("Ep:", e)
        time=0
        score = 0
        lives = 3
        done = False
        state = env.reset()
        state_mem = []
        while not done:
            t_state = np.reshape(np.pad(state,((3,3),(0,0),(0,0)), 'constant', constant_values=0),[1,216,160,3])
            action = agent.act(t_state)
            next_state, reward, done, info = env.step(action)
            t_next_state = np.reshape(np.pad(next_state,((3,3),(0,0),(0,0)), 'constant', constant_values=0),[1,216,160,3])
            
            if info['ale.lives'] < lives:
                lives-=1
                reward = -100
                
            
            agent.remember(t_state, action, reward, t_next_state,done)
            state = next_state
            time+=1
            score=max(score,score+reward)
            
            if e%1==0:
                state_mem.append(state)
                #env.render()
        
        if e%1:
            state_mem = np.save("game_"+str(e/100),np.array(state_mem))
            agent.save("spaceinvaders_inception_"+str(e/100)+".h5")
            
        with open("sp_ince_dump.csv", "a") as f:
            f.write("{}, {}, {}\n".format(e,score,time))
            
        agent.replay()
        agent.save("spaceinvaders-inception.h5")