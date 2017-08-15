import gym
import numpy as np
# Load the environment
env = gym.make('FrozenLake-v0')
# Implement Q-Table learning algorithm
# Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])      # Q table, initialized with all zeros, obs size * action size
# Set learning parameters
lr = .8
y = .95
num_episodes = 2000
# create lists to contain total rewards and steps per episode
# jList = []
rList = []
for i in range(num_episodes):
    # Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    # The Q-Table learning algorithm
    while j < 99:
        j+=1
        # Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))    # action to epsilon-greedy maxmization: max Q(s,a) + small chances of random action, epsilon diminishing
        # Get new state and reward from environment
        s1,r,d,_ = env.step(a)              # get new state, reward, destination, _ is probably steps
        # Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])       # update Q table using Q-learning algorithm  Q:= Q+ lr*TD, TD:= reward + alpha*max_a Q(new state, a) - Q(previous state, current action)
        rAll += r           # update total reward
        s = s1              # previous state set to new state
        if d == True:       # destination status
            break
    #jList.append(j)
    rList.append(rAll)