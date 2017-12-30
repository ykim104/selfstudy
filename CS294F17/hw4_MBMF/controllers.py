import numpy as np
from cost_functions import trajectory_cost_fn
import time

class Controller():
	def __init__(self):
		pass

	# Get the appropriate action(s) for this state(s)
	def get_action(self, state):
		pass


class RandomController(Controller):
	def __init__(self, env):
		""" YOUR CODE HERE """
		self.env = env
		pass

	def get_action(self, state):
		""" YOUR CODE HERE """
		""" Your code should randomly sample an action uniformly from the action space """
        
		action = np.random.uniform(self.env.action_space.low, self.env.action_space.high)

		return action


class MPCcontroller(Controller):
	""" Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
	def __init__(self, 
				 env, 
				 dyn_model, 
				 horizon=5, 
				 cost_fn=None, 
				 num_simulated_paths=10,
				 ):
		self.env = env
		self.dyn_model = dyn_model
		self.horizon = horizon
		self.cost_fn = cost_fn
		self.num_simulated_paths = num_simulated_paths

	def get_action(self, state):
		""" YOUR CODE HERE """
		""" Note: be careful to batch your simulations through the model for speed """
        
		#print("num path" + repr(self.num_simulated_paths))        
		costs = np.zeros((self.num_simulated_paths))
        
		sample_actions = np.random.uniform(self.env.action_space.low, self.env.action_space.high, (self.num_simulated_paths, self.horizon, self.env.action_space.shape[0]))
        
		#print("sample_actions" + repr(sample_actions.shape))
		#print("state" + repr(state.shape))
        
		states = np.zeros((self.num_simulated_paths, self.horizon, state.shape[0]))
		resulting_states = np.zeros((self.num_simulated_paths, self.horizon, state.shape[0]))
        
        
        # use dynamics model to associoated simulated rollouts (s_t+1 = f(s_t,a_t)
		for k in range(self.num_simulated_paths): # for timestep horizons
			actions = np.copy(sample_actions[k])
			#print(k)
			#for h in range(self.horizon):
				#print("k sample actions " + repr(actions[h].shape))
				#action = actions[h]
                # get next states
			curr_states, curr_resulting_states = self.dyn_model.predict(state, actions)
				#print("resulting_states " + repr(resulting_state.shape))  

			resulting_states[k] = curr_resulting_states
			states[k] = curr_states
				#print("resulting states append" + repr(resulting_state.shape))#state = resulting_state           
			#resulting_state = np.array(resulting_state) # 
                
        # use cost_fn to evaluate trajectories
			#print("k_accum states" + repr(states.shape))
			#print("k_accum actions" + repr(sample_actions.shape))
			#print("k_accum resulting" + repr(resulting_states.shape))
			traj_cost = trajectory_cost_fn(self.cost_fn, curr_states, sample_actions[k], curr_resulting_states)
			costs[k]=(traj_cost)
			#print("num path" + repr(self.num_simulated_paths))#all_samples(resulting_state)
            
        # find best trejectory
		best_sim_num = np.argmin(costs)
		best_seq = sample_actions[best_sim_num]
        # return first action with best trajecory
		#print("best_seq " + repr(best_sim_num))      
		action = np.copy(best_seq[0])
		#print("best_act " + repr(action.shape))
		return action