import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline

tf.reset_default_graph()
# These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[1,16],dtype=tf.float32)     # tensor input
W = tf.Variable(tf.random_uniform([16,4],0,0.01))       # weight | variable(initial_value), initial_value is a tensor with pre-specified shape
Qout = tf.matmul(inputs1,W)         # 1 layer from 16 input to 4 output
predict = tf.argmax(Qout,1)         # max Q_network for next action | argmax(input,axis=None,name=None,dimension=None)   0 for

# Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)        # next Q value for 4 actions
loss = tf.reduce_sum(tf.square(nextQ - Qout))       # summation of squared loss
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)  # set optimization method
updateModel = trainer.minimize(loss)        # set objective function


init = tf.initialize_all_variables()

# Set learning parameters
y = .99
e = 0.1
num_episodes = 2000
#create lists to contain total rewards and steps per episode
jList = []
rList = []
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        # Reset environment and get first new observation
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        # The Q-Network
        while j < 99:
            j+=1
            # Choose an action by greedily (with e chance of random action) from the Q-network
            a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(16)[s:s+1]})        # var:new_data for feeding
            # a: best action associated with max action value
            # allQ: all 4 action value

            if np.random.rand(1) < e:   # random action
                a[0] = env.action_space.sample()

            # Get new state and reward from environment
            s1,r,d,_ = env.step(a[0])       # s1: new state after action, r: reward, d: destination, _: steps

            # Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(16)[s1:s1+1]})

            #O btain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0,a[0]] = r + y*maxQ1

            # Train our network using target and predicted Q values
            _,W1 = sess.run([updateModel,W],feed_dict={inputs1:np.identity(16)[s:s+1],nextQ:targetQ})
            rAll += r
            s = s1
            if d == True:
                # Reduce chance of random action as we train the model.
                e = 1./((i/50) + 10)
                break
        jList.append(j)
        rList.append(rAll)
print "Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%"