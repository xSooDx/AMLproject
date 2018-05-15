import random
from collections import deque

import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Conv2D, Lambda, Input, concatenate, LeakyReLU, Flatten
import numpy as np

class InceptionDQAgent:
    
    def __init__(self,action_space, input_shape=(216,160,3), p_discount=0.9, n_discount=0.95, learning_rate=0.01, explore_factor=0.99, explore_decay=0.995, min_explore=0.01):
        self.action_space = action_space
        self.input_shape = input_shape
        self.epsilon = explore_factor
        self.epsilon_decay=explore_decay
        self.epsilon_min=min_explore
        self.lr = learning_rate
        self.p_gamma = p_discount
        self.n_gamma = n_discount        
        self.memory = deque(maxlen=1024)        
        self.model = self._build_model()
        
    def remember(self, state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
        
    def replay(self, batch_size = 48, sample_size = 8):
        c = 0
        for _ in range(0,len(self.memory),batch_size):
        
            states = []
            target_fs = []
            
            for _ in range(batch_size):
                data = self.memory.pop()
                state = data[0]
                action = data[1]
                target = data[2]
                next_state = data[3]
                done = data[4]
                
                
                if not done:
                    g = self.p_gamma if target>0 else self.n_gamma
                    target = (target + g*np.amax(self.model.predict(next_state)[0]))
                    
                target_f = self.model.predict(state)
                target_f[0][action] = target
                
                states.append(state)
                target_fs.append(target_f)
        
            states = np.array(states)
            target_fs = np.array(target_fs)
            
            self.model.fit(state,target_f,epochs = 1, verbose=0)
            c+=1
            if c>= sample_size:
                print("Break")
                break
                
        '''
        for state, action, reward, next_state, done in reversed(self.memory):
            target = reward
            
            if not done:
                g = self.p_gamma if reward>0 else self.n_gamma
                target = (reward + g*np.amax(self.model.predict(next_state)[0]))

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state,target_f,epochs = 1, verbose=0)
            c+=1
            if c>= batch_size:
                break
        '''    
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_vals = self.model.predict(state)
        return np.argmax(act_vals[0])
                
    def _build_model(self):
    
        def space_to_depth2(x):
            return tf.space_to_depth(x,2)
            
        def space_to_depth4(x):
            return tf.space_to_depth(x,4)
    
        L_in = Input(self.input_shape)
        
        L1_1 = Conv2D(8,21, padding="same")(L_in)
        L1_2 = Conv2D(16,15, padding="same")(L_in)
        L1_3 = Conv2D(32,7, padding="same")(L_in)
        
        L1_cat =concatenate([L1_1,L1_2,L1_3])
        L1_out = LeakyReLU()(L1_cat)
        
        L2_0 = Conv2D(32,3,padding="same")(L1_out)
        L2_1 = Lambda(space_to_depth4)(L2_0)
        L2_out = LeakyReLU()(L2_1)
        
        L3_0 = Conv2D(32,3,padding="same")(L2_out)
        L3_1 = Lambda(space_to_depth2)(L3_0)
        L3_out = LeakyReLU()(L3_1)
        
        L4_0 = Flatten()(L3_out)
        
        L4 = Dense(self.action_space, activation="linear")(L4_0)
        print(L4.shape)
        x = Model(L_in, L4) 
        x.compile("adam","mse",["accuracy"])
        
        return x
        
        
    def load(self,name):
        self.model.load_weights(name)
        
    
    def save(self,name):
        self.model.save_weights(name)
        
        
if __name__ == "__main__":
    agent = InceptionDQAgent(9)
        
