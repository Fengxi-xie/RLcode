from controller import *
import numpy as np
import socket 
import time
import pickle 
from tf2.ppo import *

STEP_TIME = 20

class DeepCar(Supervisor):

    def __init__(self):
        super(DeepCar,self).__init__()
        self.init_()
        
    def init_(self):
        #initialize robot
        wheels_name = ['motor4','motor3', 'motor1', 'motor2' ]
        self.wheels = []
        for i in range(4):
            self.wheels.append(Motor(wheels_name[i]))

        for wheel in self.wheels:
            wheel.setPosition(float('inf'))
            wheel.setVelocity(0)

        self.gps = GPS("gps")
        self.gps.enable(STEP_TIME)
        self.imu = InertialUnit("imu")
        self.imu.enable(STEP_TIME)

        self.dis_sensor = []
        for i in range(9):
            self.dis_sensor.append(DistanceSensor("ds%d"%(i)))
        for dis_ in self.dis_sensor:
            dis_.enable(STEP_TIME)

        self.touch = TouchSensor('touch')
        self.touch.enable(STEP_TIME)

        #set target zone
        self.tar = [[-2.28 , 0],
                    [-0.64 , 2.05],
                    [-4.92 , 1.58],
                    [0.61, 4.46],
                    [-4.57, 4.46],
                    [4.83, 1.77],
                    [1.78, 0.5]]

        self.xy_data = None
        self.imu_yaw= 0
        self.err_last_yaw = 0
        self.distance_data = []
        self.istouch = False
        self.dis_last = 0
        self.dis_now = 100

        self.take_once = 0
        self.time_mistake = 0

        self.target_id = 2
        self.tar_pot = None


    def enable_eqp(self):
        '''enable equipment'''
        self.gps.enable(STEP_TIME)
        # self.lidar.enable(STEP_TIME)
        self.imu.enable(STEP_TIME)
        for dis_ in self.dis_sensor:
            dis_.enable(STEP_TIME)
        self.touch.enable(STEP_TIME)


    def disable_eqp(self):
        '''disable equipment'''
        self.gps.disable()
        for dis_ in self.dis_sensor:
            dis_.disable()
        self.imu.disable()
        # self.lidar.disable()

    def refresh(self):
        '''refresh environment '''
        self.xy_data = [self.gps.getValues()[2], self.gps.getValues()[0]]
        self.imu_yaw = self.imu.getRollPitchYaw()[2]
        # print(self.imu_yaw)
        self.distance_data  = []

        for dis_ in self.dis_sensor:
            temp = dis_.getValue()
            #data nromalization
            self.distance_data.append(temp/ dis_.getMaxValue() )

        if self.touch.getValue() ==1:
            self.istouch = True
        else :
            self.istouch = False


    def limit(self, val,min,max):
        '''limit function'''
        if val < min:val = min
        elif val > max:val = max 
        return val

    def speed_set(self, l,r):
        '''set speed'''
        l = self.limit(l,-10,10)
        r = self.limit(r,-10,10)
        for wheel in self.wheels:
            wheel.setPosition(float('inf'))
   
        self.wheels[0].setVelocity(l)
        self.wheels[2].setVelocity(l)
        self.wheels[1].setVelocity(r)
        self.wheels[3].setVelocity(r)

    def pd_control(self,tar,now,err_last,k,d):
        '''pd controller'''
        err_ = (tar - now)
        out_ = k* err_+ d * (err_ - err_last)
        return out_,err_

    def setspeed_dir(self, speed_, dir_):
        '''direction control'''
        ang_w_dir, self.err_last_yaw = self.pd_control(dir_,self.imu_yaw ,self.err_last_yaw,60,2)
        l_add = - ang_w_dir
        r_add =  -l_add

        self.speed_set(speed_ + l_add ,speed_ + r_add)
        

    def direction_make(self,tar,now):
        '''use current gps and taget gps to calculate the desired angle'''
        temp  = np.array(tar) - np.array(now)
        y_ = -temp[0]
        x_ = temp[1]
        ang = np.arctan2(y_,x_)
        return ang

    def distance_make(self,tar,now):
        '''calculate speed and distance'''
        temp  = np.array(tar) - np.array(now)
        self.dis_now = np.sqrt(np.sum(np.square(temp)))
        if self.take_once ==0:
            self.dis_last = self.dis_now
            self.take_once =1

        dis_err = -(self.dis_now - self.dis_last) * 1000/STEP_TIME
        self.dis_last = self.dis_now
        return dis_err

    def reset_condition(self):
        '''determin whether the reset conditions are met'''
        # within 0.4 from the target, collision or not, whether it is beyond the boundary
        if self.dis_now < 0.4 or self.istouch or \
        self.xy_data[0] > 6.5 or self.xy_data[0] < -6.5 or\
         self.xy_data[1] > 6.5 or self.xy_data[1] < -6.5 :

            return True
        else :
            return False


    def reset(self):
        '''reset environment'''
        
        self.time_mistake -=1
        if self.time_mistake ==0 :
            #set 345 as target
            temp = np.random.randint(3)  #random target selection
            if temp == 0:self.target_id = 2
            elif temp == 1:self.target_id = 3
            elif temp == 2:self.target_id = 5
            
            self.take_once = 0
            print(self.target_id)
            print('simulationReset!!!!')


    def reward_func(self,data):
        '''reward function'''

        #here use the front, left and right sensor，
        #make sure there is no obstacle to the way forward
        dis_sensor_err = 3 - (data[3] + data[5] + data[4] )    
        rew = (2-10*dis_sensor_err) 
        return rew


robot = DeepCar()
timestep = int(robot.getBasicTimeStep())
time_count = 0
speed = 6


#define PPO
obs_dim ,act_dim,act_bound  = 10,1,1
ppo = PPO(obs_dim,act_dim,act_bound)
ep_r_all =[]


def run():
    
    ppo.load(ppo.actor,'ppo_actor')
    ppo.load(ppo.critic,'ppo_critic')

    while robot.step(timestep) != -1: 
        
        robot.step(timestep)
        robot.enable_eqp()
        robot.refresh() 
        robot.tar_pot = robot.tar[robot.target_id]   
        ang = robot.direction_make(robot.tar_pot,robot.xy_data)
        err = robot.distance_make(robot.tar_pot, robot.xy_data)  #not used

        # print(robot.tar_pot)
        if robot.reset_condition() and robot.time_mistake ==0 :
                robot.time_mistake = 2
                robot.simulationReset()
                
        if robot.time_mistake != 0 :
            robot.reset()
            continue
        
        s = robot.distance_data + [robot.imu_yaw] 
            #get action
        a = ppo.get_action(s)
        # print(a)
        #adjust the forward direction
        robot.setspeed_dir(speed,ang + float(a[0]))


def train():

    for ep in range(ppo.ep_max):

        buffer_s, buffer_a, buffer_r = [], [], []
        ep_r = 0
        
        for t in range(ppo.ep_len):  
            #refresh environment
            robot.step(timestep)
            #get state
            robot.enable_eqp()
            robot.refresh() 
            robot.tar_pot = robot.tar[robot.target_id]   
            ang = robot.direction_make(robot.tar_pot,robot.xy_data)
            err = robot.distance_make(robot.tar_pot, robot.xy_data)  #not used

            #judge whether the reset condtion are met
            if robot.reset_condition() and robot.time_mistake ==0 :
                robot.time_mistake = 2
                robot.simulationReset()
                
            if robot.time_mistake != 0 :
                robot.reset()
                continue
        
            s = robot.distance_data + [robot.imu_yaw] 
            #get action
            a = ppo.get_action(s)
            # print(a)
            #adjust the forward direction
            robot.setspeed_dir(speed,ang + float(a[0]))
            #refresh environment
            robot.step(timestep)   

            #get the next state
            robot.refresh()  
            err = robot.distance_make(robot.tar_pot, robot.xy_data)
            s_ = robot.distance_data + [robot.imu_yaw] 
            r = robot.reward_func(s_ + [err])  #err not used
            
            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append(r)

            ep_r += r

            #train after a batch size
            if (t+1) % ppo.batch_size == 0 or t == ppo.ep_len-1:
                #get reward
                v_s_ = ppo.get_value(s_)
                discounted_r = []
                for r in buffer_r[::-1]:
                    v_s_ = r + 0.9 * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()

                bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(discounted_r) #np.array(discounted_r)[:,np.newaxis]
                
                buffer_s, buffer_a, buffer_r = [], [], []
                ppo.train(bs, ba, br,ep)
        #save the parameters of net every 100 ep
        if ep % 100 ==0 and ep !=1:
            ppo.save(ppo.actor,'ppo_actor')
            ppo.save(ppo.critic,'ppo_critic')
            print('***saved***')  
            
        ep_r_all.append(ep_r)
        print(ep,'  ',ep_r)



run()
#train()
        


