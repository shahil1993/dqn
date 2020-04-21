# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 18:31:40 2018

@author: 826833
"""
import os
os.chdir(r"C:\Users\Shahil\Documents\ofcbkup\shaik\DQN")
import random
import pygame
import pandas as pd
import owncar
import imp
imp.reload(owncar)
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense,LSTM,Flatten
from keras.optimizers import Adam

EPISODES = 40


class DQNAgent:
    def __init__(self):
        self.state_size = (10,4)
        self.action_size = 2
        self.memory = deque(maxlen=2000)
        self.stateMemory=deque(maxlen=10)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.8  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_max = 1
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.decay_rate=0.01
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(LSTM(
             input_shape=self.state_size,
             units=24,
             return_sequences=True))
        model.add(Dense(24,  activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Flatten())
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        #self.memory.append((state, action, reward, next_state, done))
        self.stateMemory.append((state))
        if len(self.stateMemory)==10:
            self.memory.append((self.stateMemory,action,reward,next_state, done))
            
        

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        if len(self.stateMemory)==10:
            stary=np.array(self.stateMemory)
            stary = np.reshape(stary, [1,10, 4])
            act_values = self.model.predict(stary)
            return np.argmax(act_values[0])
        return 0
        
          # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
        #for Z in minibatch:
            target = reward
            stary=np.array(state)
            stary = np.reshape(stary, [10, 4])
            
            nstary=np.vstack([stary[1:10],next_state])
            stary = np.reshape(stary, [1,10, 4])
            nstary = np.reshape(nstary, [1,10, 4])
            
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(nstary)[0]))
            target_f = self.model.predict(stary)
            target_f[0][action] = target
            self.model.fit(stary, target_f, epochs=1, verbose=0)
    def epcng(self,episode):
        
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min)*np.exp(-self.decay_rate*(episode)) 
            #self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(name)


if __name__ == "__main__":
    #env = gym.make('MountainCar-v0')
    env = owncar.CarGame()
    
    agent = DQNAgent()
    #agent.load('model5.h5')
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32
    fd=[]
    pygame.init()
    #event= pygame.event.get()
    #model_json = model.to_json()
    #with open("model.json", "w") as json_file:
    #    json_file.write(model_json)
    
    

    for e in range(EPISODES):
        state = env.reset()
        Xp=state[2]
        state = np.reshape(state, [1, 4])
        #print(state)
        i=0
        max_state=-1
        Tr=0
        agent.save('model5.h5')
        for time in range(10000):
        #for event in pygame.event.get():
        #while(state[0][0]<0.51 and i<2000):
        #while(not done):
            i=i+1
            action = agent.act(state)
            
            R2=env.render()
            #print(R2)
               
            #env.render()
            
            if R2==0:
                action = agent.act(state)
                #action=0
            else:
                action = R2
                
            next_state, reward, done, _ = env.step(action)
            #reward = reward #if not done else -10
            next_state = np.reshape(next_state, [1, 4])
            Tr=Tr+reward
            
            #reward = reward + max(0,next_state[0][0]+0.5)
            
                
                #self.epsilon *= self.epsilon_decay
                #agent.epcng()
                
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            if done:
                break
        fd.append([e,Tr])
            #print(done)
            #if done:
            #    print("episode: {}/{}, score: {}, e: {:.2},p:{}"
            #          .format(e, EPISODES, time, agent.epsilon,state))
             #   break
       # print("episode: {}/{}, score: {}, e: {:.2},p:{}"
       #               .format(e, EPISODES, max_state, agent.epsilon,state[0][0]))
       
       

        print("E:",e," TR:",Tr, " eps:",agent.epsilon," len:",i,' XP:',Xp)       
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
            agent.epcng(e)
            
            #agent.model.save_weights("model.h5")
            #if Tr>=100:
            #    agent.epcng(e) 