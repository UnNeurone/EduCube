import numpy as np
# import tensorflow as tf
from keras.optimizers import Adam
# from tensorflow.keras.callbacks import TensorBoard
from keras.layers import Input, Conv2D, MaxPool2D, Dense, Concatenate
from keras import Model
from keras.utils import plot_model
from keras.initializers import RandomUniform
from keras.regularizers import l2
import keras
import tensorflow as tf
import random
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import gym
from math import sqrt
import time
from random import shuffle

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


class Critic(object):
    def __init__(self, Q_layer=[32, 32], activation="relu", Ad_alpha_Q=1e-3, l2_weight_decay_Q=1e-2, nb_obs=8,
                 nb_action=2):
        self._Q_model, self._opt_Q = self.build(activation=activation, Ad_alpha_Q=Ad_alpha_Q,
                                                l2_weight_decay_Q=l2_weight_decay_Q, Q_layer=Q_layer,
                                                Q_input_size=nb_obs, actions_input_size=nb_action)

    def build(self, activation, Ad_alpha_Q, l2_weight_decay_Q, Q_layer, Q_input_size, actions_input_size):
        # ------------------- Q : Critic --------------------------
        # Define Q Neural Network
        # Input layer
        #Unlike the original DDPG, the critic receives both the state and action asinput to the first layer.
        Q_input = Input(shape=(Q_input_size+actions_input_size,))

        # Setting up Q 1st layer
        f_ini = 1 / sqrt(int(Q_input.shape[1]) * Q_layer[
            0])  # 1/sqrt(fan-in) ; where fa-in is the number of incoming network connection to the system

        # Q include a L2 weight decay of 10−2
        Q = Dense(Q_layer[0], activation=activation, kernel_initializer=RandomUniform(minval=-f_ini, maxval=f_ini),
                  kernel_regularizer=l2(l2_weight_decay_Q), bias_regularizer=l2(l2_weight_decay_Q))(
            Q_input)  # 1st hidden layer

        for i in Q_layer[1:]:
            f_ini = 1 / sqrt(int(Q.shape[1]) * i)
            Q = Dense(i, activation=activation, kernel_initializer=RandomUniform(minval=-f_ini, maxval=f_ini),
                      kernel_regularizer=l2(l2_weight_decay_Q), bias_regularizer=l2(l2_weight_decay_Q))(Q)

            # Output layer
            # final layer initialized with a uniform distribution
        Q_ouput_size = 1  # final Q value
        Q_output = Dense(Q_ouput_size, activation=activation,
                         kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3),
                         kernel_regularizer=l2(l2_weight_decay_Q), bias_regularizer=l2(l2_weight_decay_Q))(Q)

        # Set-up final model
        Q_model = Model(inputs=Q_input, outputs=Q_output)
        plot_model(Q_model, "Q_model_TD3.png", show_shapes=True)
        opt = Adam(learning_rate=Ad_alpha_Q)

        lossQ = keras.losses.mean_squared_error  # Update the critic
        Q_model.compile(loss=lossQ, optimizer=opt)

        return Q_model, opt

    def get_model(self):
        return self._Q_model

    def get_opt(self):
        return self._opt_Q


class Actor(object):
    def __init__(self, mu_layer=[400, 300], activation="relu", Ad_alpha_mu=1e-4, nb_obs=8, nb_action=2):
        self._mu_model, self._opt_mu = self.build(activation, Ad_alpha_mu, mu_layer, nb_obs, nb_action)

    def build(self, activation, Ad_alpha_mu, mu_layer, mu_input_size, mu_ouput_size):
        # ------------------- mu : Actor --------------------------
        # Define mu Neural Network
        # Input layer

        mu_input = Input(shape=(mu_input_size,))  # size of the env space state

        # Setting up mu layers
        f_ini = 1 / sqrt(int(mu_input.shape[1]) * mu_layer[
            0])  # 1/sqrt(fan-in) where fa-in is the number of incoming network connection to the system
        mu = Dense(mu_layer[0], activation=activation, kernel_initializer=RandomUniform(minval=-f_ini, maxval=f_ini))(
            mu_input)
        for i in mu_layer[1:]:
            f_ini = 1 / sqrt(int(mu.shape[1]) * i)
            mu = Dense(i, activation=activation, kernel_initializer=RandomUniform(minval=-f_ini, maxval=f_ini))(mu)

            # Output layer
            # final layer initialized with a uniform distribution
            # size of the env action possible
        mu_output = Dense(mu_ouput_size, activation="tanh",
                          kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3))(mu)

        # Set-up final model
        mu_model = Model(inputs=mu_input, outputs=mu_output)
        plot_model(mu_model, "mu_model_TD3.png", show_shapes=True)
        opt_mu = Adam(learning_rate=Ad_alpha_mu)  # self.update_mu()#
        # ------------------------!-----------------------------------
        mu_model.compile(optimizer=opt_mu)  # loss='mse',
        # ------------------------!-----------------------------------
        return mu_model, opt_mu

    def get_model(self):
        return self._mu_model

    def get_opt(self):
        return self._opt_mu


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

    #def sample(self, mini_batch_size):
        #Output the MiniBatch with N(mini_batch_size) random Transitions.
    #    minibatch = random.sample(self._buffer, k=mini_batch_size)
    #    print(minibatch)
    #    return minibatch

    def sample(self, mini_batch_size):
        if len(self._buffer) < mini_batch_size:
            return [random.choice(self._buffer) for _ in range(mini_batch_size)]

        #Output the MiniBatch with N(mini_batch_size) random Transitions.
        return random.sample(self._buffer, mini_batch_size)

    #def sample(self, mini_batch_size):
    #    minibatch_index = np.random.choice(min(self._next , int(mini_batch_size)), int(self.max_size))
#
        #minibatch = self._buffer[minibatch_index]
    #    return minibatch

    def clean(self):
        self._next = 0
        self._buffer = list()

    def get_buffer(self):
        return self._buffer


# TD3 Agent class https://arxiv.org/pdf/1802.09477v3.pdf
class TD3Agent:
    def __init__(self, env='LunarLanderContinuous-v2', mu_layer=[400, 300], activation="relu", Q_layer=[400, 300],
                 Ad_alpha_mu=1e-4, Ad_alpha_Q=1e-3, l2_weight_decay_Q=1e-2, ReplayBuffer_size=1e6, MiniBatch_size=64,
                 discount_gamma=0.99, tau=0.001):

        # Environment initialization
        self.env_name = env
        self.env = gym.make(env)
        self.env.reset()
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

        #Random initialization of the 2 Critic Networks
        self.critic_value = [Q_layer,activation, Ad_alpha_Q, l2_weight_decay_Q]

        self.critic_1 = Critic(Q_layer, activation, Ad_alpha_Q, l2_weight_decay_Q, self.nb_obs, self.nb_action)
        self.Q_1 = self.critic_1.get_model()
        self.critic_2 = Critic(Q_layer, activation, Ad_alpha_Q, l2_weight_decay_Q, self.nb_obs, self.nb_action)
        self.Q_2 = self.critic_2.get_model()

        #Random initialization of the Actor Network
        self.actor_value = [mu_layer, activation, Ad_alpha_mu]
        self.actor = Actor(mu_layer, activation, Ad_alpha_mu, self.nb_obs, self.nb_action)
        self.mu = self.actor.get_model()

        #Clone the Actor and Critics to perform the soft target update to avoid divergence
        self.mu_prim = self.cloneNetwork(self.actor)
        self.Q_prim_1 = self.cloneNetwork(self.critic_1)
        self.Q_prim_2 = self.cloneNetwork(self.critic_2)

    def cloneNetwork(self, agent):
        # clone the input agent with its weights
        model_prim = keras.models.clone_model(agent.get_model())
        model_prim.compile()
        model_prim.set_weights(agent.get_model().get_weights())
        plot_model(model_prim, "clone_model.png", show_shapes=True)
        return model_prim

    def update_Qs(self, Q_1, Q_2,critic_1,critic_2, observation_t_temp, action_temp, y):
        obs_act_inputs = tf.concat([observation_t_temp, action_temp], axis=1)

        with tf.GradientTape(persistent=True) as tape:
            critic_value_1 = Q_1(obs_act_inputs, training=True)
            critic_value_2 = Q_2(obs_act_inputs, training=True)
            critic_loss_1 = tf.math.reduce_mean(tf.math.square(y - critic_value_1))
            critic_loss_2 = tf.math.reduce_mean(tf.math.square(y - critic_value_2))
            #critic_loss = critic_loss_1 +critic_loss_2

        critic_grad_1 = tape.gradient(critic_loss_1, Q_1.trainable_weights)
        critic_1.get_opt().apply_gradients(
            zip(critic_grad_1, Q_1.trainable_weights)
        ) #trainable weight instead of trainable_variables
        critic_grad_2 = tape.gradient(critic_loss_2, Q_2.trainable_weights)
        critic_1.get_opt().apply_gradients(
            zip(critic_grad_2, Q_2.trainable_weights) #trainable weight instead of trainable_variables
        )

    def update_mu(self, observation_t_temp):
        # Update the actor mu by applying the chain rule to the expected return from the start distribution

        with tf.GradientTape() as tape:
            actions = self.mu(observation_t_temp, training=True)
            obs_act_inputs = tf.concat([observation_t_temp, actions], axis=1)
            critic_value = self.Q_1(obs_act_inputs, training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.mu.trainable_weights)  #trainable_variables
        self.actor.get_opt().apply_gradients(
            zip(actor_grad, self.mu.trainable_weights)
        )



    def normalize_input(self, data):
        #Normalization of the data. It is use wih data from the MiniBatch to perform batch normalization
        data = (data - np.mean(data)) / (np.std(data))
        return data


    def target_function(self, Q_prim_1,Q_prim_2, mu_prim, MiniBatch, y,stddev_Noise):
        # Compute the target values
        # Bellman equation to minimize the loss of the Critic
        y = None
        observation_t1_temp = self.normalize_input(tf.convert_to_tensor([seq[3][0] for seq in MiniBatch], dtype='float32'))
        reward_temp = tf.convert_to_tensor([seq[2] for seq in MiniBatch], dtype='float32')

        self.Noise = np.random.normal(0, stddev_Noise, size=(self.MiniBatch_size, self.nb_action)).clip(-0.5,0.5)

        action_temp = mu_prim(observation_t1_temp, training=True)+self.Noise

        obs_act_inputs = tf.concat([observation_t1_temp, action_temp], axis=1)

        q_value_temp_1 = Q_prim_1(obs_act_inputs, training=True)
        q_value_temp_2 = Q_prim_2(obs_act_inputs, training=True)

        #y = reward_temp + self.discount_gamma * Q_prim([observation_t1_temp, action_temp], training=True)
        y = reward_temp + self.discount_gamma * np.minimum(q_value_temp_1,q_value_temp_2)
        return y

    def update_network_prim(self, Q_prim_1, Q_prim_2,mu_prim, Q_1,Q_2, mu):
        #The clone actor and clone critic weights are slowly updated with the tau factor and original networks weights
        Q_prim_1 = self.update_network_prim2(Q_1, Q_prim_1)
        Q_prim_2 = self.update_network_prim2(Q_2, Q_prim_2)

        mu_prim = self.update_network_prim2(mu, mu_prim)
        return Q_prim_1,Q_prim_2, mu_prim

    def update_network_prim2(self, model, model_prim):
        #Update of the weights of a clone from the original and the tau factor

        temp_weight_model_prim = []
        #Looping on each layer
        for layer_model, layer_model_prim in zip(model.get_weights(), model_prim.get_weights()):
            temp_weight_model_prim.append(self.tau * layer_model + (1 - self.tau) * layer_model_prim)
        model_prim.set_weights(temp_weight_model_prim)
        return model_prim

    def trainAgent(self, number_episode=50, number_timestep=100, stddev_Noise=0.2, module =2, render = True,noise_clip = 0.5):

        #The reward list will be plot at the end to see the evolution/learning of the DDPGAgent
        self.RewardList = []
        self.RewardListavg =[]

        for e in range(1, number_episode + 1):
            print("Episode : ", e)

            #Each reward episode is store. At the end of the episode, the values are sumed and added to the RewardList
            reward_episode = []

            # A new environment is set-up for each episode
            observation_t = self.env.reset()
            observation_t = tf.reshape(observation_t, [1, self.nb_obs])

            done = False #Initilaization of the state of the episode


            for t in tqdm(range(1, number_timestep + 1)):

                self.env.render() if render else False # Render or not the environment. No render increase the speed of the computation


                self.Noise = np.random.normal(0, stddev_Noise, self.nb_action).clip(-noise_clip,noise_clip)

                # the article "PARAMETER SPACE NOISE FOR EXPLORATION" show that gaussian noise is similar to Ornstein-Uhlenbeck

                # Action is outputed by the Actor with the exploration Noise
                temp_action = self.mu(observation_t)
                action = (temp_action + self.Noise)[0]

                action = np.clip(action, self.lower_bound, self.upper_bound)

                #action = (self.mu(observation_t) + self.Noise)[0]

                observation_t1, reward, done, info = self.env.step(action) # The environment perform the action and output the transition values
                observation_t1 = tf.reshape(observation_t1, [1, self.nb_obs])

                reward_episode.append(reward) # the reward is store for the later sum

                self.ReplayBuffer.push((observation_t, action, reward, observation_t1)) #Add to the buffer the new transition

                self.MiniBatch = self.ReplayBuffer.sample(self.MiniBatch_size) # Sample Minibatch of N transition from the Buffer

                self.y = self.target_function(self.Q_prim_1, self.Q_prim_2,self.mu_prim, self.MiniBatch, self.y,stddev_Noise)  # Compute the target y with the Bellman equation to minimize the loss of the Critic

                #Extract from Minibatch the observation_t and the action. Observation_t is Normalized.
                observation_t_temp = self.normalize_input(tf.convert_to_tensor([seq[0][0] for seq in self.MiniBatch],dtype="float32"))
                action_temp = np.asarray([seq[1] for seq in self.MiniBatch])

                #Update the Critics by minimizing the loss.
                self.update_Qs(self.Q_1,self.Q_2,self.critic_1,self.critic_2 , observation_t_temp, action_temp, self.y)
                #self.update_Q(self.Q_2, self.critic_2 ,observation_t_temp, action_temp, self.y)

                if t%module:
                    # Update the Actor using the policy gradient
                    self.update_mu(observation_t_temp)

                    # Clones' weights are slowly updated with the tau value
                    self.Q_prim_1,self.Q_prim_2, self.mu_prim = self.update_network_prim(self.Q_prim_1,self.Q_prim_2, self.mu_prim, self.Q_1,self.Q_2, self.mu)

                observation_t = observation_t1

                if done:
                    print("\nEpisode finished after {} timesteps".format(t + 1))
                    break

            #Total reward of one episode
            reward_episode_total = sum(reward_episode)
            print("\nReward episode : ", reward_episode_total)
            self.RewardList.append(reward_episode_total)
            self.RewardListavg.append(np.mean(self.RewardList[-40:]))

            if not(e%10):
                print("Average over last 10 episodes:", np.mean(self.RewardList[-10:]))

        #print(self.RewardList)
        #Graphic with each reward of each episode
        #plt.figure(1)
        #plt.plot(self.RewardList)
        #plt.show()

        year, month, day, hour, min = map(int, time.strftime("%Y %m %d %H %M").split())
        filepath = "./models/"+self.env_name+"_"+str(year)+"_"+str(month)+"_"+str(day)+"_"+str(hour)+"_"+str(min)+"_Keras_TD3/"

        self.Q_1.save(filepath+"_Q1")
        self.Q_2.save(filepath +"_Q2")
        self.Q_prim_1.save(filepath + "_Q1prim")
        self.Q_prim_2.save(filepath + "_Q2prim")
        self.mu.save(filepath+"_mu")
        self.mu_prim.save(filepath + "_muprim")

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
    envlist=['MountainCarContinuous-v0','LunarLanderContinuous-v2','Pendulum-v0']
    activationList = ["relu","elu"]
    Q_layer = [400,300]#[126,126,126]  #[16, 32,256]
    mu_layer =Q_layer#[256,256,256] #[256,256]

    Ad_alpha_Q = 1e-3#1e-3 #1e-3
    Ad_alpha_mu = 1e-4#1e-4 #1e-4
    tau = 0.005
    MiniBatch_size = 64#64
    discount_gamma = 0.99


    agent1 = TD3Agent(env = envlist[1], activation=activationList[0], Q_layer=Q_layer,Ad_alpha_mu=Ad_alpha_mu,Ad_alpha_Q=Ad_alpha_Q, mu_layer=mu_layer, tau=tau, MiniBatch_size=MiniBatch_size,discount_gamma =discount_gamma)
    #(self, env='LunarLanderContinuous-v2', mu_layer=[400, 300], activation="relu", Q_layer=[400, 300],Ad_alpha_mu=1e-4, Ad_alpha_Q=1e-3, l2_weight_decay_Q=1e-2, ReplayBuffer_size=1e6, MiniBatch_size=64,discount_gamma=0.99, tau=0.001)
    agent1.trainAgent(render = False,number_episode=500, number_timestep=200,stddev_Noise=0.2)
    print("--- %s seconds ---" % (time.time() - start_time))
