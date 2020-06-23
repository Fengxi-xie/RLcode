#   RL basic methods about neural networks funtion based on Tensorflow 2.0
#   Updated by Jun Wencui in 2020-5-18
#   Vision = 1.0

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp

class Net_F(layers.Layer):
    '''Define a general three-layer network'''
    def __init__(self,obs_dim,acts_dim,mult,trainable = True, name = 'Net'):
        super(Net_F,self).__init__(name = name)
        self.obs_dim = obs_dim
        self.acts_dim = acts_dim
        #Define the number of network nodes
        self.wid_mul = self.obs_dim *mult
        #Define whether to update the new parameters
        self.trainable = trainable

        #Three layers
        self.h1 = layers.Dense(self.wid_mul,activation = 'relu'
                          ,dtype = 'float64')
        self.h2 = layers.Dense(self.wid_mul,activation = 'relu'
                          ,dtype = 'float64')
        self.h3 = layers.Dense(self.acts_dim ,activation = 'linear'
                          ,dtype = 'float64')

    def call(self,input):
        #Link network
        out = self.h1(input)
        out = self.h2(out)
        out = self.h3(out)

        return out



class Actor(layers.Layer):
    '''Policy-based actor network'''
    def __init__(self,obs_dim,acts_dim,acts_bound,mult,trainable = True,name = 'Actor'):
        super(Actor,self).__init__(name = name)

        self.wid_mul = obs_dim * mult

        #Define three layers
        self.h1 = layers.Dense(self.wid_mul,activation = 'relu'
                          ,dtype = 'float64')
        self.h2 = layers.Dense(self.wid_mul,activation = 'relu'
                          ,dtype = 'float64')

        self.h2_mu = layers.Dense(self.wid_mul,activation = 'relu'
                          ,dtype = 'float64')
        #self.h2_sigma = layers.Dense(self.wid_mul,activation = 'relu'
        #                  ,dtype = 'float64')

        self.h_mu = layers.Dense(acts_dim,activation = 'tanh',dtype = 'float64')
        self.h_sigma = layers.Dense(acts_dim,activation = 'softplus',dtype = 'float64')

        self.trainable = trainable
        self.acts_bound = acts_bound

    def call(self,inputs):
        
        h1 = self.h1(inputs)
        h2 = self.h2(h1)

        h2_mu = self.h2_mu(h2)
        sigma = self.h_sigma(h2)

        mu = self.acts_bound * self.h_mu(h2_mu)
        #sigma = self.h_sigma(h2_sigma)
        norm_dist =tfp.distributions.Normal(loc = mu,scale = sigma)
  
        return norm_dist


class Critic(layers.Layer):

    '''critic network'''
    def __init__(self,obs_dim,mult,trainable = True,name = 'Critic'):
        super(Critic,self).__init__(name = name)

 
        self.wid_mul = obs_dim * mult

        self.h1 = layers.Dense(self.wid_mul,activation = 'relu'
                          ,dtype = 'float64')
        self.h2 = layers.Dense(self.wid_mul,activation = 'relu'
                          ,dtype = 'float64')
        self.h3 = layers.Dense(1,activation = 'linear'
                          ,dtype = 'float64')

        self.trainable = trainable

    def call(self,inputs):

        h1 = self.h1(inputs)
        h2 = self.h2(h1)
        v_ = self.h3(h2)
        
        return  v_




