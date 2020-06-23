import gym
import numpy as np
from draw.plot import *
from tf2.dqn import *
from tf2.ppo import *


#PPO 测试
def ppo_test():

    env = gym.make('Pendulum-v0')
    ppo = PPO(3,1,2)
    ppo.ep_max = 500

        
    for ep in range(ppo.ep_max):
        s = env.reset()
        buffer_s, buffer_a, buffer_r = [], [], []
        ep_r = 0
        for t in range(300):    # in one episode
            env.render()

            a = ppo.get_action(s)

            s_, r, done, _ = env.step(a)
            s_ = np.squeeze(s_)
        
            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append((r+2)/2)    # normalize reward, find to be useful
  
            s = s_
            ep_r += r

            if (t+1) % ppo.batch_size == 0 or t == 300-1:
        
                v_s_ = ppo.get_value(s_)
            
                discounted_r = []
                for r in buffer_r[::-1]:
                    v_s_ = r + 0.9 * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()
          
                bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(discounted_r) #np.array(discounted_r)[:,np.newaxis]
            
                buffer_s, buffer_a, buffer_r = [], [], []
                ppo.train(bs, ba, br,ep)

        ep_r_all.append(ep_r)
        print(ep,'  ',ep_r)

    plot(ppo.a_loss)
    plot(ppo.c_loss)
    plot(ep_r_all)


