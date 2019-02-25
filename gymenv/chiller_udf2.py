import logging
import numpy as np
import random
from gym import spaces
import gym
from gym.utils import seeding
from os import path
import tensorflow as tf
import pdb

logger = logging.getLogger(__name__)

class ChillerEnv_2(gym.Env):
    def __init__(self):
        self.cof = 0.01  ## scale of CL AND SP
        self.num = 3 ## inputs numbers
        self.dir = '/home/paperspace/Documents/Chiller/Actor-Critic-Process-control/MLPparameters/epoches_18.npz'
        r = np.load('/home/paperspace/Documents/Chiller/Actor-Critic-Process-control/Data/Datasets.npz')
        self.orginalData = r['arr_0']        
        self.CLrange = np.array([946.71, 1945.83])
        self.SPrange = np.array([805.4,1201.038])
        self.CHWRTrange = np.array([10.42,13.84])
        self.CWSTrange = np.array([26.07,30.82])
        MLP_parameters = np.load(self.dir)
        self.w1 = MLP_parameters["arr_3"]
        self.w2 = MLP_parameters["arr_4"]
        self.b1 = MLP_parameters["arr_5"]
        self.b2 = MLP_parameters["arr_6"]
        self.count = 0
        ##weights and biases
    def getGamma(self):
        return self.gamma
    def getStates(self):
        return self.state    
    def calInputs(self,increment):
        #pdb.set_trace()
        self.inputs_renor = self.inputs_renor + increment
        return self.inputs_renor    
    def step(self,action):        
        #系统当前的状态
        #pdb.set_trace()        
        state = self.state
        state_value = self.orginalData[state,0] ##~1000
        #pdb.set_trace()
        # action
        #print(action)
        action_value1 = action[0]*((self.CHWRTrange[1]-self.CHWRTrange[0])/2)+(self.CHWRTrange[0]+self.CHWRTrange[1])/2
        action_value2 = action[1]*((self.CWSTrange[1]-self.CWSTrange[0])/2)+(self.CWSTrange[0]+self.CWSTrange[1])/2
        SP = self.getRewards(np.hstack((state_value*self.cof,action_value1,action_value2)))##about state
        #pdb.set_trace()
        rewards_nor = -SP/(state_value*self.cof)
        #rewards_nor = (rewards-(9.4671+19.4583)/2)/((19.4583-9.4671)/2)-1
        #self.inputs_renor = self.calInputs(action_renor)
        next_state = self.state +1        
        self.state = next_state        
        ter = False        
        return self._get_obs(),rewards_nor,ter,{}
    def getRewards(self,inputs):
        #pdb.set_trace()
        inputs = np.reshape(inputs,(1,self.num))
        inputs = inputs.astype(np.float32)
        temp1 = tf.matmul(inputs,self.w1)+self.b1;
        self.h1= tf.nn.sigmoid(temp1)
        #pdb.set_trace()
        self.h2 = tf.matmul(self.h1,self.w2)+self.b2;        
        #pdb.set_trace()
        return self.h2.eval()
    def currentState(self):
        self.state_sequential = np.array([self.state-12,self.state-6,self.state])
        state_value = [self.orginalData[self.state-12,0],self.orginalData[self.state-6,0],self.orginalData[self.state,0]]
        #print(state_value)
        state_value_nor =state_value
        for i in range(len(state_value)):
            state_value_nor[i] = (state_value[i]-(self.CLrange[0]+self.CLrange[1])/2)/((self.CLrange[1]-self.CLrange[0])/2)
        return np.reshape(state_value_nor,(3,1))
    
    def reset(self):
        ## CHWS_temp and CWS_temp        
        #CWS_temp = random.uniform(-1.0,1.0)
        ## objective load
        self.state = 13 + self.count
        self.count = self.count + 6 
        #Normalized CHWS temp
        #self.inputs_renor = np.array([CHWS_temp])
        ###actual CHWS temp
        #self.inputs_renor = np.array([(self.inputs[0]*((self.CHWSrange[1]-self.CHWSrange[0])/2)+(self.CHWSrange[0]+self.CHWSrange[1])/2)])
        #state = self.getState(self.inputs)        
        #self.state = state[0][0] ## CL,SP,rated_CL
        #pdb.set_trace()
        print(self.state)
        print("reset_state:",self.orginalData[self.state,0]*0.01)
        #
        sq = self.currentState()
        print('sequential:',sq)
        return sq,self._get_obs()
    def _get_obs(self):
        state_value = self.orginalData[self.state,0] ###~1000
        ##normalization_CL
        state_value_nor = (state_value - (self.CLrange[0]+self.CLrange[1])/2)/((self.CLrange[1]-self.CLrange[0])/2)
        ### the state achieved is a normalized state
        return np.array([state_value_nor])
    
