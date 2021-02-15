import time

import gym
import pyglet
from gym import error, spaces, utils
from gym.utils import seeding, EzPickle
from scipy import signal
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)
import numpy as np
from gym.envs.classic_control import rendering
from tqdm import tqdm
from math import sqrt, pi, exp
#inspired from https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py
from os import path
import matplotlib.pyplot as plt
FPS = 50
SCALE = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

HEATERS_POWER = 1
LIGHT_POWER = 1

INITIAL_RANDOM = 1000.0

PLANT_POLY =[
    (-14, +17), (-17, 0), (-17 ,-10),
    (+17, -10), (+17, 0), (+14, +17)
    ]

VIEWPORT_W = 600
VIEWPORT_H = 400

class PlantObject():
    def __init__(self, full_life = 200, max_size = 50,speed_grow = 0.4, color = 'green', mu = 0.5, sig = 1):
      self.age = 0
      self.full_life = full_life
      self.max_size = max_size
      self.speed_grow = speed_grow
      self.color = 'green'
      self.size = 0
      self.alive = 0

      #Reaction of the plant to the parameter
      self.mu = mu
      self.sig = sig

    def nextState(self, light, temp):
      self.age += 1

      if self.age> self.full_life:
        self.alive = 1
      else:
        self.size = (self.gaussian(light) + self.gaussian(temp))*self.speed_grow + self.size
        if light>0.75 or temp>0.75:
          self.age+=50

        if self.size > self.max_size:
          self.size = self.max_size

      return self.size, self.age, self.alive

    def getState(self):
      return self.size, self.age, self.alive

    def getSize(self):
      return self.size

    def getMaxSize(self):
      return self.max_size

    def gaussian(self, input):
      return np.exp(-np.power(input - self.mu, 2.) / (2 * np.power(self.sig, 2.)))

    def reset(self):
      self.age = 0
      self.size = 0
      self.alive = 0

class EduCubePlantEnv(gym.Env, EzPickle):
    metadata = {'render.modes': ['human', 'rgb_array'],'video.frames_per_second': FPS}
    def __init__(self):
      EzPickle.__init__(self)
      self.seed()
      self.viewer = None

      self.world = Box2D.b2World()

      #Object env definition
      self.plant1 = PlantObject()
      self.plant1env = None
      self.plant2 = PlantObject()
      self.plant2env = None
      self.plant3 = PlantObject()
      self.plant3env = None
      self.plant4 = PlantObject()
      self.plant4env = None
      self.plant5 = PlantObject()
      self.plant5env = None
      self.plant6 = PlantObject()
      self.plant6env = None

      self.plantlist = [self.plant1,self.plant2,self.plant3,self.plant4,self.plant5,self.plant6]

      self.heater1 = None
      self.heater2 = None
      self.heater3 = None
      self.heater4 = None
      self.heater5 = None
      self.heater6 = None

      self.heaterlist = [self.heater1,self.heater2,self.heater3,self.heater4,self.heater5,self.heater6]

      self.light = None

      self.prev_reward = None

      self.nb_obs_state = 6*3 + 6 + 1
      self.observation_space = spaces.Box(0, np.inf, shape=(self.nb_obs_state,), dtype=np.float32)
      self.previous_state = []

      self.action_space = spaces.Box(0, +1, (7,), dtype=np.float32)

      self.plant_r_list = []
      self.sizelist =len(self.plantlist)
      self.leef_r_list = [[] for _ in range(0,self.sizelist)] #[list()] * len(self.plantlist)

      self.firstframe =True
      self.reset()

    def seed(self, seed=None):
      self.np_random, seed = seeding.np_random(seed)
      return [seed]

    def reset(self):
      self.plant1.reset()
      self.plant2.reset()
      self.plant3.reset()
      self.plant4.reset()
      self.plant5.reset()
      self.plant6.reset()

      W = VIEWPORT_W / SCALE
      H = VIEWPORT_H / SCALE

      self.plant1env = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
      self.plant1env.color1 = (0.0, 255, 0.0)
      self.firstframe = True
      self.previous_state =[]
# self.moon.CreateEdgeFixture(vertices=[p1,p2],density=0,friction=0.1)
      self.rgb = 5
      self.plant_r_list = []
      self.leef_r_list = [[] for _ in range(0,self.sizelist)]

      return self.step(np.array([0,0,0,0,0,0,0]))[0]

    def step(self, action):
      action = np.clip(action, -1, +1).astype(np.float32)
      state = []
      state_temp_previous =[]
      reward = 0
      done = False
      temp_done = 0

      #action = heater1,heater2,heater3,heater4,heater5,heater6, light
      for i in range (0,len(self.plantlist)):
        size,age,alive = self.plantlist[i].nextState(action[-1],action[i])
        self.heaterlist[i] = action[i]
        state.extend([size,age,alive,action[i]])
        state_temp_previous.append([size,age,alive,action[i]])

        if len(self.previous_state)!=0:
          if alive == 0:
            reward += (size - self.previous_state[i][0])/100#size - self.previous_state[i][0]
            reward += age/100#age
          else :
            reward -= 100
            temp_done+=1
        else:
          reward += 10

      state.append(action[-1])
      self.previous_state = state_temp_previous

      if temp_done == 6:
        done = True

      return np.array(state, dtype=np.float32), reward, done, {}

    def getSize(self):
      return [self.plant1.getSize(),self.plant2.getSize(),self.plant3.getSize(),self.plant4.getSize(),self.plant5.getSize(),self.plant6.getSize()]

    def render(self, mode='human'):
      width_plant, width_plant_temp = 3, 3
      space_between, space_between_temp = 3, 1
      size_tube = 12
      bottom_start = 3.5
      fname = path.join(path.dirname(__file__), "tubesv2.jpg")
      nb_leef = 4
      if self.viewer is None:
        self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
        self.viewer.set_bounds(0, VIEWPORT_W / SCALE, 0, VIEWPORT_H / SCALE)

        self.image = rendering.Image(fname, VIEWPORT_W/15, VIEWPORT_H/15)
        self.viewer.add_geom(self.image)

      else:
        if self.firstframe:
          for plant in self.plantlist:
            size, age, alive = plant.getState()
            max_size = plant.getMaxSize()
            l, r, t, b = space_between, width_plant, bottom_start + size / 10, bottom_start  # -50 / 2, 50 / 2, 30 / 2, -30 / 2
            space_between += width_plant_temp / 1.25
            width_plant += width_plant_temp

            plant_body, plant_leef = self.makeBody(l + 0.75, b, size, max_size, nb_leef)
            plant_r = rendering.FilledPolygon(plant_body)
            plant_r.set_color(0, 255, 0)

            self.viewer.add_geom(plant_r)
            self.plant_r_list.append(plant_r)
            self.firstframe = False
        else:
          for plant in range(0,len(self.plantlist)):
            size, age, alive = self.plantlist[plant].getState()
            max_size = self.plantlist[plant].getMaxSize()
            l, r, t, b = space_between,width_plant,bottom_start+size/10,bottom_start#-50 / 2, 50 / 2, 30 / 2, -30 / 2
            space_between += width_plant_temp/1.25
            width_plant += width_plant_temp

            plant_body, plant_leef  = self.makeBody(l+0.75,b, size, max_size,nb_leef)

            temp_plant = self.plant_r_list[plant]
            temp_plant.v = plant_body
            if alive == 0:
              temp_plant.set_color(0,255,0)
            else:
              temp_plant.set_color(0, 0,0)

            for leef in range(0,len(plant_leef)):
              #print("plant :", plant)
              #print("A :", len(self.leef_r_list[plant]),"| B :",len(plant_leef))
              if len(self.leef_r_list[plant]) < len(plant_leef):
                #print("plant :", plant)
                #print("leef :", leef)
                #print(plant_leef[leef])
                leef_r = rendering.FilledPolygon(plant_leef[leef])
                leef_r.set_color(0,255,0)
                self.viewer.add_geom(leef_r)
                self.leef_r_list[plant].append(leef_r)
              else:
                temp_leef = self.leef_r_list[plant][leef]
                temp_leef.v = plant_leef[leef]
                if alive == 0:
                  temp_leef.set_color(0, 255, 0)
                else:
                  temp_leef.set_color(0, 0, 0)


          #plant_r = rendering.FilledPolygon(plant_body)
          #plant_r.set_color(0,255,0)




      return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def leefPoly(self, start_x, start_y,size_leef=0.25):
      return [(start_x, start_y), (start_x+size_leef, start_y+size_leef), (start_x+size_leef*2, start_y+size_leef), (start_x+size_leef*3, start_y),(start_x+size_leef*2, start_y-size_leef),(start_x+size_leef, start_y-size_leef)]

    def mainBody(self,start_x, start_y,size_body_width=0.4,size_body_height=0.4):
      return [(start_x, start_y),(start_x, start_y+size_body_height),(start_x+size_body_width, start_y+size_body_height),(start_x+size_body_width, start_y)]

    def makeBody(self,start_x, start_y, size, max_size,nb_leef =4):
      nb_leef_percent = (size/max_size)*100
      size_body_width = 0.4
      size_body_height = size/10
      size_body_height_step = 0.5
      size_leef = 0.25
      mainBodyPlant = self.mainBody(start_x, start_y,size_body_width,size_body_height)
      listLeef = []

      temp_start_x = start_x - size_leef*3
      temp_start_y = start_y + size_body_height_step/2 + size_body_height/4
      max_leef = round((nb_leef_percent+(100/nb_leef))/(100/nb_leef))

      for i in range(0,max_leef-1):
        if (i+1)%2 == 1:
          leef = self.leefPoly(temp_start_x, temp_start_y,size_leef)
          temp_start_x = temp_start_x +size_leef*3+size_body_width
          temp_start_y = temp_start_y + size_body_height_step + size_body_height/8#+ size_body_height/2
        else:
          leef = self.leefPoly(temp_start_x, temp_start_y, size_leef)
          temp_start_x = temp_start_x - size_leef * 3 - size_body_width
          temp_start_y = temp_start_y + size_body_height_step
        listLeef.append(leef)

      return mainBodyPlant,listLeef

    def close(self):
      if self.viewer is not None:
        self.viewer.close()
        self.viewer = None


if __name__ == '__main__':
        env = EduCubePlantEnv()
        for episode in tqdm(range(50)):
          env.reset()
          for _ in range(200):
            env.render()
            #print(env.getSize())
            obs, reward, done, info = env.step(env.action_space.sample())  # take a random action
            time.sleep(0.2)
            if done == True:
              break
          env.close()