import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tf2.net import Actor,Critic
from tf2.misc import update_tar_var,learing_rate_adjust

class Actor_M(tf.keras.Model):

    def __init__(self,obs_dim,acts_dim,acts_bound,mult,name = 'ppo_actor'):
        super(Actor_M,self).__init__(name = name)

        self.obs_dim = obs_dim
        self.acts_dim = acts_dim

        self.actor = Actor(obs_dim,acts_dim,acts_bound,mult,trainable = True,name = 'pi')
        self.actor_old = Actor(obs_dim,acts_dim,acts_bound,mult,trainable =False,name = 'old_pi')
        
    def call(self,inputs,chose):

        if chose =='pi':
            pi = self.actor(inputs)
            return pi

        elif chose== 'all':
            pi = self.actor(inputs)
            pi_old = self.actor_old(inputs)

            return pi,pi_old


class Critic_M(tf.keras.Model):

    def __init__(self,obs_dim,mult,name = 'ppo_critic'):
        super(Critic_M,self).__init__(name = name)

        self.obs_dim = obs_dim
        self.critic = Critic(obs_dim,mult,trainable = True,name = 'critic')

    def call(self,inputs):
        return self.critic(inputs)



class PPO():
    def __init__(self,
                 obs_dim,
                 acts_dim,
                 act_bound,
                 lr_rate_a = 0.0001,
                 lr_rate_c = 0.0002,
                 batch = 32,
                 mult = 50,
                 ep_length = 200):

        tf.keras.backend.set_floatx('float64')  #64-bit backend operation

        self.obs_dim = obs_dim
        self.acts_dim = acts_dim
        self.batch_size = batch
        self.act_bound = act_bound
        self.gmma = 0.9
        self.ep_len = ep_length
        self.ep_max= 3000
        self.mult = mult
        self.lr_rate_a = lr_rate_a
        self.lr_rate_c = lr_rate_c

        self.discount_reward = None

        self.actor = Actor_M(obs_dim,acts_dim,self.act_bound,self.mult)

        self._update_weights()
        
        self.critic = Critic_M(obs_dim,self.mult)

        self.opt_a = tf.keras.optimizers.Adam(learning_rate=lr_rate_a)
        self.opt_c = tf.keras.optimizers.Adam(learning_rate=lr_rate_c)
        
        self.a_loss = []
        self.c_loss = []

        self.ppo_reload = False

        print('ppo build')

    def train(self,state,action,dis_reward,ep_count):

        state = tf.convert_to_tensor(np.array(state)) 
        action = tf.convert_to_tensor(np.array(action)) 
        dis_reward = tf.convert_to_tensor(np.array(dis_reward))
        #Check the data format
        # print(state)
        #print(action)
        #print(dis_reward)
        
        lr_now = learing_rate_adjust(0.00012,0.00004,self.ep_max,ep_count,300)
        
        if self.lr_rate_a != lr_now:
            self.opt_a.learning_rate = lr_now
            self.opt_c.learning_rate = 2*lr_now       
            print('The actor learning rate is ', lr_now)

        self.lr_rate_a = lr_now      

        if dis_reward.ndim != 2:
            raise ValueError('dis_reward has shape problem!')
            
        with tf.GradientTape() as tape2:

            adv_,c_loss = self._error_cal_c(state,dis_reward)
            c_grads = tape2.gradient(c_loss,self.critic.trainable_variables)
            self.opt_c.apply_gradients(zip(c_grads,self.critic.trainable_variables))

        with tf.GradientTape() as tape:

            a_loss = self._error_cal_a(state,action,adv_)
            a_grads = tape.gradient(a_loss,self.actor.actor.trainable_variables)
            self.opt_a.apply_gradients(zip(a_grads,self.actor.actor.trainable_variables))


        if self.ppo_reload:
            self.ppo_reload = False
            self.load(self.actor,'./ppo_actor')
            self.load(self.critic,'./ppo_critic')

        
            return 

        #Actor and new parameters
        self._update_weights()
     
        self.a_loss.append(a_loss.numpy())
        self.c_loss.append(c_loss.numpy())

    @tf.function
    def _error_cal_c(self,state,dis_reward):
        '''Calculate critic loss'''
        #critic loss
        v_ = self.critic(state)
        adv_ = dis_reward - v_
        adv_mean = tf.reduce_mean(tf.square(adv_))

        return adv_,adv_mean

    #@tf.function
    def _error_cal_a(self,state,action,adv_):
        '''Calculate actor loss'''
        #Actor loss
        pi,oldpi = self.actor(state,'all')
        
        ratio = pi.prob(action)/oldpi.prob(action)

        pg_losses   = ratio * adv_
        pg_losses2 = tf.clip_by_value(ratio, 1.-0.2, 1.+ 0.2)*adv_
        action_loss = -tf.reduce_mean(tf.minimum(pg_losses,pg_losses2))

        return  action_loss

    def get_value(self,state):
        '''Get a reference from the critic'''
        state = np.array(state)
        state = np.expand_dims(state,axis =0)
        if state.ndim !=2:
            raise ValueError('state has shape problem!')

        v_ = self.critic(tf.constant(state))[0]
        return v_.numpy()

    def get_action(self,state):
        '''get the action'''
        state = np.array(state)
        state= state[np.newaxis,:]
        if state.ndim !=2:
            raise ValueError('state has shape problem!')
        action = self._get_action(tf.constant(state))
        
        return action.numpy()

    @tf.function
    def _get_action(self,state):
        '''get the action'''
        action_probs = self.actor(state,'pi')   
        
        return tf.clip_by_value(action_probs.sample(1)[0],
                                -self.act_bound ,self.act_bound )


    def _update_weights(self):
        '''Update Expectation Strategy'''
        update_tar_var(self.actor.actor_old.weights,self.actor.actor.weights)
        pass

    def load(self, target, name):  #only save model
        target.load_weights(name)

    def save(self,target, name):
        target.save_weights(name)




