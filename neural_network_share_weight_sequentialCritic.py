""" 
Define neural network structures of the actor and critic method

The actor and critic networks share the layers: State ==> FC ==> ReLU ==> Feature

The algorithm is tested on the Pendulum-v0 OpenAI gym task 

Author: Shusen Wang
"""
import tensorflow as tf
import numpy as np
import pdb


class NeuralNetworks:
    '''
    State_Feature: State to Feature
    Action_Feature: Action to Feature
    Actor: State_Feature to Action
    Critic: (State_Feature, Action_Feature) to Value
    '''
    
    def __init__(self, sess, state_dim, action_dim, action_bound):
        # size of layers
        self._S_FEATURE_DIM = 2
        self._ACTOR_H1_DIM = 4
        self._CRITIC_H1_DIM = 4
        self.sess =sess
        # constants
        self._S_DIM = state_dim
        self._A_DIM = action_dim
        self._A_BOUND = action_bound

        self._S_DIM_c = 3
        self._A_DIM_c = 2
        criticparams = np.load('./Data/criticparams_nor.npz')
        self.criticparams = criticparams['arr_0']
        actorparams = np.load('./Data/actorparams_nor.npz')
        self.actorparams = actorparams['arr_0']
        #pdb.set_trace()
        # Create actor network
        # features extracted from states
        #import pdb
        #pdb.set_trace()
        #self.input_state, self.state_feature = self._create_state_feature(sigma=0.3) ##x,w,b==>y
        #import pdb        
        #param_state_feature = tf.trainable_variables() ##2:w;b
        # actor network
        #pdb.set_trace()
        self.input_state, self.state_feature = self._create_state_feature(sigma=0.3)
        param_state_feature = tf.trainable_variables()
        self.actor_y = self._create_actors(self.state_feature)###########
        # parameters of actor network
        self.actor_params = tf.trainable_variables()
        #pdb.set_trace()
        num_params1 = len(tf.trainable_variables())
        
     
        
        # Create critic network
        # features extracted from states and actions
        self.input_state_critic = self.input_state   ##current x
        self.state_feature_critic = self.state_feature   ##current y
        self.input_action = tf.placeholder(tf.float32, [None, self._A_DIM])
        # critic network
        self.critic_y = self._create_critics(self.state_feature_critic, self.input_action)###########
        # parameters of critic network
        self.critic_params = param_state_feature +  tf.trainable_variables()[num_params1:] ############start from num_params2
        num_params3 = len(tf.trainable_variables())
        
        
           
        
    def get_const(self):
        return self._S_DIM, self._A_DIM, self._A_BOUND
    
    def get_input_state(self, is_target=False):
        if is_target:
            return self.input_state_target
        else:
            return self.input_state
    
    def get_actor_out(self, is_target=False):
        if is_target:
            return self.actor_target_y
        else:
            #pdb.set_trace()
            
            return self.actor_y
        
    def get_actor_params(self, is_target=False):
        if is_target:
            return self.actor_target_params
        else:
            return self.actor_params
        
    
    def get_input_state_action(self, is_target=False):
        if is_target:
            return (self.input_state_critic_target, self.input_action_target)
        else:
            return (self.input_state_critic, self.input_action)
    
    def get_critic_out(self, is_target=False):
        if is_target:
            return self.critic_target_y
        else:
            return self.critic_y
        
    def get_critic_params(self, is_target=False):
        if is_target:
            return self.critic_target_params
        else:
            return self.critic_params
        
    
    
    # ================ Shared Functions ================ #
    def weight_variable(shape, value=0.1, rand='normal'):
        if rand == 'normal':
            initial = tf.truncated_normal(shape, stddev=value)
        elif rand == 'uniform':
            initial = tf.random_uniform(shape, minval=-value, maxval=value)
        return tf.Variable(initial)

    def bias_variable(shape, value=0.1):
        initial = tf.constant(value, shape=shape)
        return tf.Variable(initial)
    
    # =========== Define Networks Structures =========== #
    def _create_state_feature(self, sigma=0.1):
        '''
        State ==> FC ==> ReLU ==> Feature
        '''
        #pdb.set_trace()
        x = tf.placeholder(tf.float32, [None, self._S_DIM])        
        w = tf.Variable(self.criticparams[0]) ## initial weights: random produce [3,64] float32
        b = tf.Variable(self.criticparams[1])
        y = tf.nn.relu(tf.matmul(x, w) + b)
        return x, y
    
    def _create_actors(self, feature):
        '''
        State_Feature ==> FC ==> ReLU ==> FC ==> Tanh ==> Scale ==> Action
        '''
        #pdb.set_trace()
        #feature = tf.placeholder(tf.float32, [None, self._S_DIM]) 
        w1 = tf.Variable(self.actorparams[0]) 
        b1 = tf.Variable(self.actorparams[1])
        h1 = tf.nn.relu(tf.matmul(feature, w1) + b1)
        w2 = tf.Variable(self.actorparams[2])
        b2 = tf.Variable(self.actorparams[3])
        y = tf.nn.tanh(tf.matmul(h1, w2) + b2)
        
        return y
    
        
    def _create_critics(self, s_feature, action):
        '''
        Hidden Layer: 
            H1 = State_Feature * W1s + Action * W1a + Bias
        Critic:
            H1 ==> ReLU ==> FC ==> Value
        '''
        #pdb.set_trace()
        w1_s = tf.Variable(self.criticparams[2])
        w1_a = tf.Variable(self.criticparams[3])
        b1 = tf.Variable(self.criticparams[4])
        h1 = tf.add(tf.matmul(s_feature, w1_s), tf.matmul(action, w1_a))
        y1 = tf.nn.relu(tf.add(h1, b1))
        w2 = tf.Variable(self.criticparams[5])
        b2 = tf.Variable(self.criticparams[6])
        h2 = tf.nn.tanh(tf.matmul(y1, w2) + b2)
        y2 = tf.add(h2,-1)
        return y2
        
