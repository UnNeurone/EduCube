import numpy as np
# import tensorflow as tf
#from keras.optimizers import Adam
# from tensorflow.keras.callbacks import TensorBoard
#from keras.layers import Input, Conv2D, MaxPool2D, Dense, Concatenate
#from keras import Model
#from keras.utils import plot_model
#from keras.initializers import RandomUniform
#from keras.regularizers import l2
#import keras
#import tensorflow as tf
import random
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import gym
from math import sqrt
import time
import torch
import torch.nn  as nn
import torch.optim as optim
import copy
from EduCubePlant_env_v4 import EduCubePlantEnv
import torch.nn.functional as F

from random import shuffle

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


class Critic(nn.Module):
    def __init__(self, Q_layer, nb_obs, nb_action):
        super(Critic, self).__init__()

        self.ln1 = nn.Linear(nb_obs,Q_layer[0])
        f_ini = 1 / sqrt(int(nb_obs * Q_layer[0]))  # 1/sqrt(fan-in) where fa-in is the number of incoming network connection to the system
        nn.init.uniform_(self.ln1.weight,-f_ini,f_ini)

        self.ln2 = nn.Linear(Q_layer[0]+nb_action,Q_layer[1])
        f_ini = 1 / sqrt(int((Q_layer[0]+nb_action) * Q_layer[1]))  # 1/sqrt(fan-in) where fa-in is the number of incoming network connection to the system
        nn.init.uniform_(self.ln2.weight,-f_ini,f_ini)

        self.ln3 = nn.Linear(Q_layer[1],1)
        nn.init.uniform_(self.ln3.weight, -3e-3, 3e-3)

    def forward(self, x1, x2):
        x = torch.relu(self.ln1(x1))
        x = torch.cat([x, x2], dim=1)
        x = torch.relu(self.ln2(x))
        x = self.ln3(x)
        return x


class Actor(nn.Module):
    def __init__(self, mu_layer, nb_obs, nb_action):
        super(Actor, self).__init__()

        self.ln1 = nn.Linear(nb_obs,mu_layer[0])
        f_ini = 1 / sqrt(int(nb_obs * mu_layer[0]))  # 1/sqrt(fan-in) where fa-in is the number of incoming network connection to the system
        nn.init.uniform_(self.ln1.weight,-f_ini,f_ini)

        self.ln2 = nn.Linear(mu_layer[0],mu_layer[1])
        f_ini = 1 / sqrt(int(mu_layer[0] * mu_layer[1]))  # 1/sqrt(fan-in) where fa-in is the number of incoming network connection to the system
        nn.init.uniform_(self.ln2.weight,-f_ini,f_ini)

        self.ln3 = nn.Linear(mu_layer[1],nb_action)
        nn.init.uniform_(self.ln3.weight, -3e-3, 3e-3)

    def forward(self, x1):
        x = torch.relu(self.ln1(x1))
        x = torch.relu(self.ln2(x))
        x = torch.sigmoid_(self.ln3(x))
        return x

class ReplayBuffer:
    def __init__(self, size):
        self.max_size = size
        self._next = 0
        self._buffer = list()

    def push(self, elem):

        if len(self._buffer) < self.max_size:
            self._buffer.append(elem)
            self._next = self._next +1
        else:
            #When the buffer is full, it discards older transition
            self._buffer[self._next] = elem
            self._next = (self._next + 1) % self.max_size

    def sample(self, mini_batch_size):
        if len(self._buffer) < mini_batch_size:
            return [random.choice(self._buffer) for _ in range(mini_batch_size)]

        #Output the MiniBatch with N(mini_batch_size) random Transitions.
        return random.sample(self._buffer, mini_batch_size)

    def clean(self):
        self._next = 0
        self._buffer = list()

    def get_buffer(self):
        return self._buffer


# DDPG Agent class
class DDPGAgent:
    def __init__(self, env='LunarLanderContinuous-v2', mu_layer=[400, 300], Q_layer=[400, 300],
                 Ad_alpha_mu=1e-4, Ad_alpha_Q=1e-3, l2_weight_decay_Q=1e-2,ReplayBuffer_size=1e6, MiniBatch_size=64,
                 discount_gamma=0.99, tau=0.001):

        # Environment initialization
        self.env = EduCubePlantEnv()#gym.make(env)
        self.env.reset()
        self.env_name = env
        self.nb_obs = self.env.observation_space.shape[0]
        self.nb_action = self.env.action_space.shape[0]

        self.upper_bound = self.env.action_space.high[0]
        self.lower_bound = self.env.action_space.low[0]

        # Other Variable
        self.Noise = None  # Ornstein-Uhlenbeck process for exploration but PARAMETER SPACE NOISE FOR EXPLORATION show that gaussian noise is similar in result (correlated)
        self.ReplayBuffer = ReplayBuffer(ReplayBuffer_size)
        self.MiniBatch_size = MiniBatch_size
        self.MiniBatch = []
        self.RewardList = []

        # Target
        self.y = None

        # self.MiniBatch = [None] * MiniBatch_size
        self.discount_gamma = discount_gamma
        self.tau = tau

        #Random initialization of the Critic Network
        self.critic_value = [Q_layer, Ad_alpha_Q, l2_weight_decay_Q]
        self.critic = Critic(Q_layer, self.nb_obs, self.nb_action)
        self.critic_opt = optim.Adam(self.critic.parameters(),lr = Ad_alpha_Q, weight_decay=l2_weight_decay_Q)

        #Random initialization of the Actor Network
        self.actor_value = [mu_layer, Ad_alpha_mu]
        self.actor = Actor(mu_layer, self.nb_obs, self.nb_action)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=Ad_alpha_mu)

        #Clone the Actor and Critic to perform the soft target update to avoid divergence
        self.mu_prim = copy.deepcopy(self.actor)
        self.Q_prim = copy.deepcopy(self.critic)

    def update_Q(self, observation_t_temp, action_temp, y):
        output = self.critic(observation_t_temp,action_temp)
        criterion = nn.MSELoss()
        loss = criterion(output, y)
        loss.backward()
        self.critic_opt.step()
        self.critic_opt.zero_grad()

    def update_mu(self, observation_t_temp):
        # Update the actor mu by applying the chain rule to the expected return from the start distribution
        loss =  -self.critic(observation_t_temp, self.actor(observation_t_temp)).mean()
        loss.backward()
        self.actor_opt.step()
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()


    def normalize_input(self, data):
        #Normalization of the data. It is use wih data from the MiniBatch to perform batch normalization
        data = (data - np.mean(data)) / (np.std(data))
        return data


    def target_function(self):
        # Compute the target values
        # Bellman equation to minimize the loss of the Critic
        self.y = None
        observation_t1_temp = torch.tensor([seq[3] for seq in self.MiniBatch],dtype=torch.float)
        #observation_t1_temp = torch.from_numpy(self.normalize_input([seq[3] for seq in self.MiniBatch])).float()
        reward_temp = torch.tensor([float(seq[2]) for seq in self.MiniBatch])
        action_temp = self.mu_prim(observation_t1_temp)
        q_value_temp = self.Q_prim(observation_t1_temp,action_temp)#.squeeze().tolist()
        self.y = reward_temp + self.discount_gamma * q_value_temp


    def update_network_prim(self):
        #The clone actor and clone critic weights are slowly updated with the tau factor and original networks weights
        self.update_network_prim2(self.critic,self.Q_prim)
        self.update_network_prim2(self.actor,self.mu_prim)


    def update_network_prim2(self, model, model_prim):
        #Update of the weights of a clone from the original and the tau factor
        temp_weight_model_prim = []
        #Looping on each layer
        for layer_model, layer_model_prim in zip(model.parameters(), model_prim.parameters()):
            layer_model_prim.data.copy_(self.tau * layer_model.data + (1 - self.tau) * layer_model_prim.data)

    def trainAgent(self, number_episode=50, number_timestep=100, stddev_Noise = 0.2, render = True):

        #The reward list will be plot at the end to see the evolution/learning of the DDPGAgent
        self.RewardList = []
        self.RewardListavg =[]

        for e in range(1, number_episode + 1):
            print("Episode : ", e)

            #Each reward episode is store. At the end of the episode, the values are sumed and added to the RewardList
            reward_episode = []

            # A new environment is set-up for each episode
            observation_t = self.env.reset()

            done = False #Initilaization of the state of the episode

            for t in tqdm(range(1, number_timestep + 1)):

                self.env.render() if render else False # Render or not the environment. No render increase the speed of the computation

                self.Noise = torch.from_numpy(np.random.normal(0, stddev_Noise, self.nb_action))
                # the article "PARAMETER SPACE NOISE FOR EXPLORATION" show that gaussian noise is similar to Ornstein-Uhlenbeck

                # Action is outputed by the Actor with the exploration Noise
                temp_action = self.actor(torch.from_numpy(observation_t).float())
                #temp_action = self.actor(torch.from_numpy(self.normalize_input(observation_t)).float())
                action = (temp_action + self.Noise)

                action = action.clip(self.lower_bound, self.upper_bound).detach().numpy()

                observation_t1, reward, done, info = self.env.step(action) # The environment perform the action and output the transition values

                reward_episode.append(reward) # the reward is store for the later sum

                self.ReplayBuffer.push((observation_t, action, reward, observation_t1)) #Add to the buffer the new transition

                self.MiniBatch = self.ReplayBuffer.sample(self.MiniBatch_size) # Sample Minibatch of N transition from the Buffer

                self.target_function()  # Compute the target y with the Bellman equation to minimize the loss of the Critic

                #Extract from Minibatch the observation_t and the action. Observation_t is Normalized.
                observation_t_temp = torch.tensor([seq[0] for seq in self.MiniBatch],dtype=torch.float)
                #observation_t_temp = torch.from_numpy(self.normalize_input([seq[0] for seq in self.MiniBatch])).float()
                action_temp = torch.tensor([seq[1] for seq in self.MiniBatch],dtype=torch.float)

                #Update the Critic by minimizing the loss.
                self.update_Q(observation_t_temp, action_temp, self.y)

                #Update the Actor using the policy gradient
                self.update_mu(observation_t_temp)

                # Clones' weights are slowly updated with the tau value
                self.update_network_prim()

                observation_t = observation_t1

                if done:
                    print("\nEpisode finished after {} timesteps".format(t + 1))
                    break
            self.env.close()
            #Total reward of one episode
            reward_episode_total = sum(reward_episode)
            print("\nReward episode : ", reward_episode_total)
            self.RewardList.append(reward_episode_total)
            self.RewardListavg.append(np.mean(self.RewardList[-2:]))


        year, month, day, hour, min = map(int, time.strftime("%Y %m %d %H %M").split())
        filepath = "./models/"+self.env_name+"_"+str(year)+"_"+str(month)+"_"+str(day)+"_"+str(hour)+"_"+str(min)+"_Pytorch_DDPG/"
        if not os.path.isdir(filepath):
            os.makedirs(filepath)
        torch.save(self.critic.state_dict(), filepath+"critic_1.pth")
        torch.save(self.actor.state_dict(), filepath + "actor.pth")
        torch.save(self.Q_prim.state_dict(), filepath + "Q_prim_1.pth")
        torch.save(self.mu_prim.state_dict(), filepath + "mu_prim.pth")

        filetext = open(filepath+"parameters.txt","w+")
        filetext.write("Critic \n")
        filetext.write(', '.join(str(e) for e in self.critic_value)+"\n")
        filetext.write("Actor \n")
        filetext.write(', '.join(str(e) for e in self.actor_value)+"\n")
        filetext.write("tau : " + str(self.tau) + ", minibatch_size : "+ str(self.MiniBatch_size) + ", discount_gamma : " + str(self.discount_gamma))
        filetext.write("\nnb episode : " + str(number_episode) + ", nb timestep : " + str(number_timestep))
        filetext.close()

        print(self.RewardListavg)
        #Graphic with each reward of each episode average
        plt.plot(self.RewardList)
        plt.plot(self.RewardListavg)
        plt.savefig(filepath+"RewardList_and_Avg.png")
        plt.show(block=False)
        plt.pause(10)
        plt.close()




if __name__ == '__main__':
    start_time = time.time()
    envlist=['MountainCarContinuous-v0','LunarLanderContinuous-v2']
    activationList = ["relu","elu"]
    Q_layer = [400,300] #[16, 32,256]
    mu_layer = Q_layer#[400,300] #[256,256]

    Ad_alpha_Q = 1e-3#5e-3#1e-3 #1e-3
    Ad_alpha_mu = 1e-4#1e-2 #1e-4 #1e-4

    tau = 0.001
    MiniBatch_size = 64
    discount_gamma = 0.99

    agent1 = DDPGAgent(env = "EduCubePlantEnv", Q_layer=Q_layer,Ad_alpha_mu=Ad_alpha_mu,Ad_alpha_Q=Ad_alpha_Q, mu_layer=mu_layer, tau=tau, MiniBatch_size=MiniBatch_size,discount_gamma =discount_gamma)
    #(self, env='LunarLanderContinuous-v2', mu_layer=[400, 300], activation="relu", Q_layer=[400, 300],Ad_alpha_mu=1e-4, Ad_alpha_Q=1e-3, l2_weight_decay_Q=1e-2, ReplayBuffer_size=1e6, MiniBatch_size=64,discount_gamma=0.99, tau=0.001)
    agent1.trainAgent(render = True,number_episode=20, number_timestep=100,stddev_Noise=0.2)
    print("--- %s seconds ---" % (time.time() - start_time))
