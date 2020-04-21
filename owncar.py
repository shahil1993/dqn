# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 15:34:11 2018

@author: 826833
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 29 16:34:30 2018

@author: 826833
"""
import pygame
import time
import random
import os
import numpy as np

os.chdir(r"C:\Users\Shahil\Documents\ofcbkup\shaik\DQN")
#pygame.init()
class CarGame:
    def __init__(self):
        self.display_width = 300
        self.display_height = 600
        self.thing_width = 100
        self.thing_height = 100
        self.black = (0,0,0)
        self.white = (255,255,255)
        self.red = (255,0,0)
        self.force_mag = 10.0
        self.tau = 0.02 
        self.car_width=50
        self.R1=0
        self.a=5
        
    def reset(self):
        self.thing_startx = random.randrange(0, self.display_width)
        
        self.thing_starty = 0
        self.thing_speed = 1
        self.x = (self.display_width * 0.45)
        #self.x =15
        self.y = (self.display_height * 0.8)
        self.state = (self.thing_startx,self.thing_starty,self.x,self.y)
        #self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        #self.steps_beyond_done = None
        return np.array(self.state)
    def step(self,action):
        state=self.state
        TX,TY,X,Y=state
        if action ==0:
            X_change=self.a
        #elif action==1:
        #    X_change=5
        elif action==1:
            if self.a==5:
                self.a=-5
                X_change=self.a
            else:
                self.a=5
                X_change=self.a
        elif action==2:
            X_change=-5
        X+=X_change
        TY += self.thing_speed
        done=False
        reward=0
        if X > self.display_width -self.car_width:
            reward=-1
            X=self.display_width -self.car_width
        if X<0:
            reward=-1
            X=0
       
            #reward=1
            #X_change=0
            #done=True
        #X+=X_change
        if TY > self.display_height:
            reward=100
            TX=random.randrange(0, self.display_width)
            TY=0
            self.thing_speed += 0.1
        if Y < TY+self.thing_height:
            if X > TX and X < TX + self.thing_width or X+self.car_width > TX and X + self.car_width < TX+self.thing_width:
                reward=-100
                done=True
        self.state=(TX,TY,X,Y)
        return np.array(self.state), reward, done, {}
            
        

   
    
    def render(self):
        state=self.state
        TX,TY,X,Y=state
        pygame.init()
        gameDisplay = pygame.display.set_mode((self.display_width,self.display_height))
        clock = pygame.time.Clock()
        gameDisplay.fill(self.white)
        
        #R1=0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN :
                if event.key == pygame.K_LEFT:
                    self.R1=2
                if event.key == pygame.K_RIGHT:
                    self.R1=1

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                    self.R1=0
        carImg = pygame.image.load('racecar1.png')
        gameDisplay.blit(carImg,(X,Y))
        #pygame.draw.rect(gameDisplay,self.red, [X,Y , 5, 5])
        pygame.draw.rect(gameDisplay,self.black, [TX, TY, self.thing_width, self.thing_height])
        pygame.display.update()
        return self.R1
        #time.sleep(2)
        #pygame.quit()
        #pygame.display.update()
        #clock.tick(60)
        
"""     
    
if __name__ == "__main__":
    #game_loop()
    #pygame.quit()
    #quit()
    

  

  

    

    

    
    

    
        
  
black = (0,0,0)
white = (255,255,255)
agent=CarGame()

state=agent.reset()
agent.render()
TX,TY,X,Y=state
pygame.init()
gameDisplay = pygame.display.set_mode((800,600))
gameDisplay.fill(white)
clock = pygame.time.Clock()
pygame.draw.rect(gameDisplay,black, [300,10, 100,100])
carImg = pygame.image.load('racecar1.png')
gameDisplay.blit(carImg,(X,Y))

pygame.display.update()
clock.tick(60)
pygame.quit()
"""

        