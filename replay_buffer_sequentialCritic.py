""" 
Data structure for implementing experience replay
"""
from collections import deque
import random
import numpy as np
import pdb

class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences 
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t):
        experience = (s, a, r, t)        
        if self.count < self.buffer_size: 
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
        #print(self.buffer)
        #pdb.set_trace()

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []
        #import pdb
        
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)
        #print (batch)
        #pdb.set_trace()# find next five sentences aim
        s_batch_show = np.array([_[0] for _ in batch])
        #print (s_batch_show)
        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])        
        #print (s_batch)
        #pdb.set_trace()
        return s_batch, a_batch, r_batch, t_batch  

    def clear(self):
        self.deque.clear()
        self.count = 0


