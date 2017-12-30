import tensorflow as tf
import numpy as np
import math

# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder, 
              output_size,
              scope, 
              n_layers=2, 
              size=500, 
              activation=tf.tanh,
              output_activation=None
              ): 
    out = input_placeholder
    with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

class NNDynamicsModel():
    def __init__(self, 
                 env, 
                 n_layers,
                 size, 
                 activation, 
                 output_activation, 
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """
        # init vars
        self.env = env
        self.n_layers = n_layers
        self.size = size
        self.activation = activation
        self.output_activation = output_activation
        self.mean_obs = normalization[0]
        self.std_obs = normalization[1]
        self.mean_delta = normalization[2]
        self.std_delta = normalization[3]
        self.mean_act = normalization[4]
        self.std_act = normalization[5]
        self.batch_size = batch_size
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.sess = sess
        
        #placeholders
        #self.inputs_placeholder = tf.placeholder(tf.float32, shape=[None, size], name='input') 
        #self.labels_placeholder = tf.placeholder(tf.float32, shape=[None, size], name='output') 
        
        self.foo = 0
    def fit(self, data):
        self.foo +=1
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """   
        
        """YOUR CODE HERE """
        # normalize dataset
        l_obs = data['observations']
        l_next_obs = data['next_observations']
        l_deltas = l_next_obs - l_obs
        l_act = data['actions']
        
        
        e = np.random.normal(0,.1)
        norm_l_obs = np.nan_to_num((l_obs - self.mean_obs)/(e + self.std_obs))
        norm_l_deltas = np.nan_to_num((l_deltas - self.mean_delta)/(e + self.std_delta))
        norm_l_acts = np.nan_to_num((l_act - self.mean_act)/(e + self.std_act))
        
        #print("obs " + repr(norm_l_obs.shape))
        #print("acts " + repr(norm_l_acts.shape))
        #print("deltas " + repr(norm_l_deltas.shape))
        
        inputs = np.concatenate((norm_l_obs, norm_l_acts), axis=1)
        labels = np.copy(l_deltas)
        
        # placeholders
        
        inputsize = inputs.shape[1]
        outputsize = labels.shape[1]
        #print("input size = " + repr(inputsize))
        #print(outputsize)
        
        inputs_placeholder = tf.placeholder(tf.float32, shape=[None, inputsize], name='inputs') 
        labels_placeholder = tf.placeholder(tf.float32, shape=[None, outputsize], name='outputs')
        
        # feedforward NN 
        self.sess.run(tf.global_variables_initializer())
        curr_nn_output = build_mlp(inputs_placeholder, outputsize, repr(self.foo))
        #tf.get_variable_scope().reuse_variables()


        # loss
        mse = tf.reduce_mean(tf.square(labels_placeholder - curr_nn_output))
        
        # Compute gradients and update parameters
        loss = tf.reduce_mean(tf.square(mse - labels_placeholder))
        opt = tf.train.AdamOptimizer(self.learning_rate)
        theta = tf.trainable_variables()
        #tf.get_variable_scope().reuse_variables()
        gv = [(g,v) for g,v in
                    opt.compute_gradients(mse, theta)
                    if g is not None]
        train_step = opt.apply_gradients(gv)
        
        # initialize all values
        self.sess.run(tf.global_variables_initializer())
        
        # get loss from dataset 
        avg_loss = 0
        num_batch = 0 
        nData = inputs.shape[0]
        for batch in range(int(math.floor(nData/self.batch_size))):
                              
            inputs_batch = inputs[batch*self.batch_size:(batch+1)*self.batch_size, :]
            labels_batch = labels[batch*self.batch_size:(batch+1)*self.batch_size, :]
                              
           
            _, a_loss , output, true_ouotput = self.sess.run([train_step, mse, curr_nn_output, labels_placeholder],
                               feed_dict={inputs_placeholder: inputs_batch, labels_placeholder: labels_batch})
            avg_loss += a_loss
            num_batch += 1
        
        #print(self.batch_size)
        #print(nData)
        
        return avg_loss/num_batch

                                                         
                                                         
    def predict(self, state, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """
        """ YOUR CODE HERE """
  
        #print("predict_states " + repr(state.shape))
        #print("predict_actions " + repr(actions.shape))
        
        next_states = np.zeros((actions.shape[0], state.shape[0]))
        states = np.copy(next_states)
        
        #print("input_list" + repr(input_list.shape[0]))
        
        next_state = state
        for n_roll in range(actions.shape[0]):
            states[n_roll] = state
            input_list = np.concatenate((next_state, actions[n_roll]))
            #print("input_shape " + repr(input_list.shape))
            inputs = np.reshape(input_list, (1, input_list.shape[0]))
        
            # placeholder
            inputsize = input_list.shape[0]
            outputsize = state.shape[0]
            inputs_placeholder = tf.placeholder(tf.float32, shape=[None,inputsize], name='inputs') 
        
            # feedforward NN
            curr_nn_output = build_mlp(inputs_placeholder, outputsize, "scope")
        
            # initialize all values        
            self.sess.run(tf.global_variables_initializer())
            output = self.sess.run([curr_nn_output], feed_dict={inputs_placeholder: inputs})
            state_delta = np.multiply(output, self.std_delta) + self.mean_delta
            #print(state_delta.shape)
            state_deltas = np.reshape(state_delta, (state_delta.shape[2]))
            #print(state_deltas.shape)
            #print(inputs.shape)
            state += state_deltas 
            next_states[n_roll] = state
            #next_states.append(next_state)
            
        #print("predict_next_states " + repr(np.array(next_states).shape))
        #print(next_states.shape)
        return states, next_states
        
