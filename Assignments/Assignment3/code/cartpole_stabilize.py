import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt

def compute_advantage(j, r, gamma):
    ### Part f) Advantage computation
    """ Computes the advantage function from data
        Inputs:
            j     -- list of time steps 
                    (eg. j == [0, 1, 2, 3, 0, 1, 2, 3, 4, 5] means that there 
                     are two episodes, one with four time steps and another with
                     6 time steps)
            r     -- list of rewards from episodes corresponding to time steps j
            gamma -- discount factor
        
        Output:
            advantage -- vector of advantages correponding to time steps j
    """
    subList=[r[0]]
    subAdvantage=[]
    advantage=[]
    length = len(j)
    for i in range(1, length):
        if j[i]!=0:
            subList.append(r[i])
        if i==length-1 or j[i+1]==0:
            running_add=0
            for t in reversed(range(len(subList))):
                running_add = running_add * gamma + subList[t]
                subAdvantage.insert(0, running_add)
            advantage+=subAdvantage
            subList=[]
            subAdvantage=[]
        if j[i]==0:
            subList.append(r[i])

    mean = np.mean(advantage)
    std = np.std(advantage)
    advantage = (advantage - mean)/std
    np.array(advantage)

    return (advantage,mean)

class agent():
    def __init__(self, lr, s_size, a_size, h1_size, h2_size):
        """ Initialize the RL agent 
        Inputs:
            lr      -- learning rate
            s_size  -- # of states
            a_size  -- # of actions (output of policy network)
            h1_size -- # of neurons in first hidden layer of policy network
            h2_size -- # of neurons in second hidden layer of policy network
        """
        self.sess = tf.Session()
        # Data consists of a list of states, actions, and rewards
        self.state_data = tf.placeholder(shape=[None,s_size], dtype=tf.float32)
        self.action_data = tf.placeholder(shape=[None], dtype=tf.int32)        
        self.advantage_data = tf.placeholder(shape=[None], dtype=tf.float32)
        ### --- Part c) Define the policy network ---
        # Input should be the state (defined above)
        # Output should be the probability distribution of actions

        # fc1
        fc1 = tf.layers.dense(
            inputs=self.state_data,
            units=h1_size,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # fc2
        fc2 = tf.layers.dense(
            inputs=fc1,
            units=h2_size,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )
        all_act = tf.layers.dense(
            inputs=fc2,
            units=a_size,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='all_act'
        )
        ### -----------------------------------------
        
        ### -- Part d) Compute probabilities of realized actions (from data) --
        # Indices of policy network outputs (which are probabilities) 
        # corresponding to action data
        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

        ### -------------------------------------------------------------------
        
        ### -- Part e) Define loss function for policy improvement procedure --
        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.action_data)   # this is negative log of chosen action
            self.loss = tf.reduce_mean(neg_log_prob * self.advantage_data)  # reward guided loss
        ### -------------------------------------------------------------------
        
        # Gradient computation
        tvars = tf.trainable_variables()
        self.gradients = tf.gradients(self.loss, tvars)
        
        # Apply update step
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradients, tvars))
    
    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.state_data: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action
    # --- end def ---
# --- end class ---

##### Main script #####
env = gym.make('CartPole-v0') # Initialize gym environment
gamma = 0.99                  # Discount factor

# initialize tensor flow model
tf.reset_default_graph()

### --- Part g) create the RL agent ---
# myAgent = agent(...)  
myAgent = agent(lr=0.02, a_size=env.action_space.n, s_size=env.observation_space.shape[0], h1_size= 8, h2_size=8)
# uncomment fill in arguments of the above line to initialize an RL agent whose
# policy network contains two hidden layers with 8 neurons each

### -----------------------------------

total_episodes = 2500 # maximum # of training episodes
max_steps = 500 # maximum # of steps per episode (overridden in gym environment)
update_frequency = 5 # number of episodes between policy network updates

# Begin tensorflow session
init = tf.global_variables_initializer()   
with tf.Session() as sess:
    # Initialization
    sess.run(init)
    i = 0
    
    ep_rewards = []
    history = []
    Means=[]
    while i < total_episodes:
        # reset environment
        s = env.reset() 

        for j in range(max_steps):
            # Visualize behaviour every 100 episodes
            if i % 100 == 0:
                env.render()
            # --- end if ---
            
            ### ------------ Part g) -------------------
            ### Probabilistically pick an action given policy network outputs.
            
            prob_weights=sess.run(myAgent.all_act_prob, feed_dict={myAgent.state_data: s[np.newaxis, :]})
            a = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
            ### ----------------------------------------
            
            # Get reward for taking an action, and store data from the episode
            s1,r,d,_ = env.step(a) #
            history.append([j,s,a,r,s1])
            s = s1

            if d == True: # Update the network when episode is done
                # Update network every "update_frequency" episodes
                if i % update_frequency == 0 and i != 0:
                    # Compute advantage
                    history = np.array(history)
                    advantage, mean= compute_advantage(history[:,0], history[:,3], gamma)
                    Means.append(mean)
                    ### --- Part g) Perform policy update ---
                    sess.run(myAgent.update_batch, feed_dict={
                        myAgent.state_data: np.vstack(history[:,1]),
                        myAgent.action_data: np.array(history[:,2]),
                        myAgent.advantage_data: advantage
                    })
                    ### -------------------------------------
                    # self.state_data = tf.placeholder(shape=[None,s_size], dtype=tf.float32)
                    # self.action_data = tf.placeholder(shape=[None], dtype=tf.int32)        
                    # self.advantage_data = tf.placeholder(shape=[None], dtype=tf.float32)
                    
                    # Reset history
                    history = []
                # --- end if ---
                break
            # --- end if ---
        # --- end for ---

        i += 1
    # --- end while ---

plt.plot(Means)
plt.ylabel('sum of discounted reward')
plt.show()
# --- end of script ---