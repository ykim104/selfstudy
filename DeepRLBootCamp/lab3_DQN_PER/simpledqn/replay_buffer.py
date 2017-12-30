"""
This project was developed by Rein Houthooft, Rocky Duan, Peter Chen, Pieter Abbeel for the Berkeley Deep RL Bootcamp, August 2017. Bootcamp website with slides and lecture videos: https://sites.google.com/view/deep-rl-bootcamp/.

Code adapted from OpenAI Baselines: https://github.com/openai/baselines

Copyright 2017 Deep RL Bootcamp Organizers.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""


import numpy as np
import random
import pickle

from segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer(object):
    def __init__(self, max_size):
        """Simple replay buffer for storing sampled DQN (s, a, s', r) transitions as tuples.

        :param size: Maximum size of the replay buffer.
        """
        self._buffer = []
        self._max_size = max_size
        self._idx = 0

    def __len__(self):
        return len(self._buffer)

    def add(self, obs_t, act, rew, obs_tp1, done):
        """
        Add a new sample to the replay buffer.
        :param obs_t: observation at time t
        :param act:  action
        :param rew: reward
        :param obs_tp1: observation at time t+1
        :param done: termination signal (whether episode has finished or not)
        """
        data = (obs_t, act, rew, obs_tp1, done)
        if self._idx >= len(self._buffer):
            self._buffer.append(data)
        else:
            self._buffer[self._idx] = data
        self._idx = (self._idx + 1) % self._max_size

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._buffer[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of transition tuples.

        :param batch_size: Number of sampled transition tuples.
        :return: Tuple of transitions.
        """
        idxes = [random.randint(0, len(self._buffer) - 1)
                 for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def dump(self, file_path=None):
        """Dump the replay buffer into a file.
        """
        file = open(file_path, 'wb')
        pickle.dump(self._buffer, file, -1)
        file.close()

    def load(self, file_path=None):
        """Load the replay buffer from a file
        """
        file = open(file_path, 'rb')
        self._buffer = pickle.load(file)
        file.close()

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        """
        Prioritied Experience Replay 
      
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha > 0
        self._alpha = alpha
        
        # I don't understand purpose of this
        # maybe to create a graph to store ranked truples? 
        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2 
        
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
    
    def add(self, *args, **kwargs):
        idx = self._idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha
        
    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self._buffer) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res
    
    def sample(self, batch_size, beta):
        assert beta > 0
       

        idxes = self._sample_proportional(batch_size)
        
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._buffer)) ** (-beta)
        
        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._buffer)) ** (-beta)
            weights.append( weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])
    
    def update_priorities(self, idxes, priorities):
        """
        set priority of transition at index idxes[i] in buffer to priorities[i]
        """
        
        
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0<= idx < len(self._buffer)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha
            
            self._max_priority = max(self._max_priority, priority)
            
          
        
        