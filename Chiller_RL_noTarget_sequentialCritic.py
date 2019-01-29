import gym
import tensorflow as tf
import numpy as np
from gym import wrappers
from gym.envs.registration import register
from neural_network_share_weight_sequentialCritic import NeuralNetworks
from replay_buffer_sequentialCritic import ReplayBuffer
import pdb
import random
import matplotlib.pyplot as plt

# ==========================
#   Training Parameters
# ==========================
# Max training steps
MAX_EPISODES = 10780#12000
# Max episode length
MAX_EP_STEPS = 200#1000
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.00001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.0001
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001
total_reward=np.zeros((10000,1))
# ===========================
#   Utility Parameters
# ===========================
# Render gym env during training
RENDER_ENV = False
# Use Gym Monitor
GYM_MONITOR_EN = False
# Gym environment
ENV_NAME = 'Chiller-v2'
# Directory for storing gym results
MONITOR_DIR = './results/gym_chiller_seCri'
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/tf_chiller_seCri'
# File for saving reward and qmax
RESULTS_FILE = './results/rewards_chiller_seCri.npz'
REWARD_FILE = './results/total_rewards_chiller_seCri.npz'
Parameters_File = './results/actor_critic_parameters_seCri.npz'
RANDOM_SEED = 123
# Size of replay buffer
BUFFER_SIZE = 50000#600
MINIBATCH_SIZE = 1#64#128

## register Chiller in gym
register(
    id = 'Chiller-v2',
    entry_point = 'gym.envs.classic_control:ChillerEnv_2',
    max_episode_steps=200,
    reward_threshold=1
    )

class Actor():
    
    def __init__(self, sess, network, learning_rate):
        self.sess = sess
        self.learning_rate = learning_rate
        _, self.a_dim, _ = network.get_const()
        
        self.inputs = network.get_input_state(is_target=False)
        self.out = network.get_actor_out(is_target=False)
        self.params = network.get_actor_params(is_target=False)
        # This gradient will be provided by the critic network
        self.critic_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients
        self.policy_gradient = tf.gradients(tf.multiply(self.out, -self.critic_gradient), self.params)
        
        # Optimization Op        
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.policy_gradient, self.params))
        
    def train(self, state, c_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: state,
            self.critic_gradient: c_gradient
        })

    def predict(self, state):
        return self.sess.run(self.out, feed_dict={
            self.inputs: state
        })
    def show_params(self,state):
        return self.sess.run(self.params,feed_dict={
            self.inputs: state,
        })   

        
class Critic:
    def __init__(self, sess, network, learning_rate):
        self.sess = sess
        self.learning_rate = learning_rate

        # Create the critic network
        self.state, self.action = network.get_input_state_action(is_target=False)
        self.out = network.get_critic_out(is_target=False)
        self.params = network.get_critic_params(is_target=False)
        
        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        #self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.loss = tf.nn.l2_loss(self.predicted_q_value - self.out)        
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.action_grads = tf.gradients(self.out, self.action)        

    def train(self, state, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.state: state,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, state, action):
        return self.sess.run(self.out, feed_dict={
            self.state: state,
            self.action: action
        })
        
    def action_gradients(self, state, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: state,
            self.action: actions
        })
    

def train(sess, env, network):
    arr_reward = np.zeros(MAX_EPISODES)
    arr_qmax = np.zeros(MAX_EPISODES)

    actor = Actor(sess, network, ACTOR_LEARNING_RATE)    
    critic = Critic(sess, network, CRITIC_LEARNING_RATE)    
    
    s_dim, a_dim, _ = network.get_const()    
    
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
    
    #actor_target.train()
    #critic_target.train()    
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)
    k=0
    dataset=[]
    ep_last_reward = -10000.
    ep_ave_max_q_last = -10.

    for i in range(MAX_EPISODES):

        s_sequential,s = env.reset()        
        ep_reward = 0
        ep_ave_max_q = 0
        #pdb.set_trace()
        
        #print('s:',s)
        #print('s_sequential:',s_sequential)
        #print(s_sequential[0])
        #a_2 = actor.predict(np.reshape(s_sequential[0], (1, 1))) + (1. / (1. + i))*0.01
        #a_1 = actor.predict(np.reshape(s_sequential[1], (1, 1))) + (1. / (1. + i))*0.01

        for j in range(MAX_EP_STEPS):            

            if RENDER_ENV:
                env.render()
            a = actor.predict(np.reshape(s_sequential, (1, s_dim))) + (1. / (1. + i))
            
            #print('a:',a)
            
            #a_sequential = np.array([a_2,a_1,a])
            s2, r, terminal, info = env.step(a[0])
            #
            #pdb.set_trace()
            replay_buffer.add(np.reshape(s_sequential, (s_dim,)), np.reshape(a, (a_dim,)), r,
                              terminal)

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() >= MINIBATCH_SIZE:
                s_batch, a_batch, r_batch, t_batch= \
                    replay_buffer.sample_batch(MINIBATCH_SIZE)
                #import pdb
                #pdb.set_trace()
                
                # Calculate targets
                #target_q = critic_target.predict(s2_batch, actor_target.predict(s2_batch))

                y_i = []
                flagggg = 1
                for k in range(MINIBATCH_SIZE):
                    #if t_batch[k]:
                        #import pdb
                        #pdb.set_trace()
                    y_i.append(r_batch[k])
                    #else:
                        #y_i.append(r_batch[k] + GAMMA * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))

                #ep_ave_max_q += np.amax(predicted_q_value)
                ep_ave_max_q += np.mean(predicted_q_value) #average of minibatch

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                #print(grads)
                #pdb.set_trace()
                actor.train(s_batch, grads[0])
                
                #import pdb
                #pdb.set_trace()

                # Update target networks
                #actor_target.train()
                #critic_target.train()
                #batch_params,batch_mean,batch_var = actor.batch_nor(s_batch)
                #batch_params_target,batch_mean_target,batch_var_target = \
                                                  #actor_target.batch_nor(s_batch)

           
            s = s2
            old_s = s_sequential
            #pdb.set_trace()
            s_sequential = np.hstack((old_s[1],old_s[2],s))
            #print("s_sequential:",s_sequential)
            #a_2 = a_1
            #a_1 = a
            
            ep_reward += r          
           
        if  replay_buffer.size() >= MINIBATCH_SIZE:
            #origin_state =s_batch[0]*((19.4583-9.4671)/2)+(9.4671+19.4583)/2
            #origin_action =  a_batch[0]*((8.00-6.70)/2)+((8.00+6.70)/2)
            print('r[k]:'+str(r_batch[0]))
            print('Real: '+str(y_i[0]))
            #print('target_q:'+str(target_q[0:20]))
            print('predicted: '+str(predicted_q_value[0]))
            print('State:'+str(s_batch[0]))
            print('action:'+str(a_batch[0]))
        
        print('Reward_average: ' + str(ep_reward/float(j+1)) + ', steps ,'\
                      + str(j) + ', Episode: ' + str(i) + \
                      ', Qmax_average: ' +  str(ep_ave_max_q / float(j+1)))

        total_reward[i,0]=ep_reward/float(j+1)
        arr_reward[i] = ep_reward
        arr_qmax[i] = ep_ave_max_q / float(j+1)
        np.savez(RESULTS_FILE, arr_reward[0:i], arr_qmax[0:i], j)
        np.savez(REWARD_FILE, total_reward)
        #plt.plot(total_reward)

        actor_params = sess.run(actor.params)       
        critic_params = sess.run(critic.params)       
        if ep_last_reward < ep_reward:
            ep_last_reward = ep_reward
            np.savez('./results/critic_notargetsq.npz',critic_params)
        if ep_ave_max_q_last < ep_ave_max_q:
            ep_ave_max_q_last = ep_ave_max_q
            np.savez('./results/actor_notargetsq.npz',actor_params)
        np.savez(Parameters_File, str(actor_params),str(critic_params))
        #np.savez(Parameters_File, str(actor_params),\
                             #str(actor_target_params),str(critic_params),\
                             #str(critic_target_params),str(batch_mean),\
                 #str(batch_var),str(batch_mean_target),str(batch_var_target),\
                 #str(batch_params),str(batch_params_target))        


    
def main(_):
 global sess
 with tf.Session() as sess:
    env=gym.make(ENV_NAME)
    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)
    env.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    sess.run(tf.global_variables_initializer())

    state_dim = 3    #状态维度
    action_dim = 2   #行动维度
    action_bound = [[4.,5.]] #最大行动值

    network = NeuralNetworks(sess, state_dim, action_dim, action_bound)
   
    if GYM_MONITOR_EN:
            if not RENDER_ENV:
                env = wrappers.Monitor(
                    env, MONITOR_DIR, video_callable=False, force=True)
            else:
                env = wrappers.Monitor(env, MONITOR_DIR, force=True)
        #import pdb
    #pdb.set_trace()    
    train(sess, env, network)
    if GYM_MONITOR_EN:
        env.monitor.close()

if __name__ == '__main__':
    tf.app.run()


