import gym
import gym_hyrja
import time
import numpy as np
import tensorflow as tf

def set_move(val):
    if val==1:
        return 1
    else:
        return -1
#
# env
tf.reset_default_graph()
env = gym.make("Hyrja-v0")
obs = env.reset()
# param
inputs1 = tf.placeholder(shape=[1,187],dtype=tf.float32)     # tensor input
W1 = tf.Variable(tf.random_uniform([187,80],-0.1,0.1))
W2 = tf.Variable(tf.random_uniform([80,50],-0.1,0.1))
W3 = tf.Variable(tf.random_uniform([50,32768],-0.1,0.1))
layer1_out=tf.matmul(inputs1,W1)
layer2_out=tf.matmul(layer1_out,W2)
Qout=tf.matmul(layer2_out,W3)
# action to max Q(s,a)
predict = tf.argmax(Qout,1)         # max Q_network for next action | argmax(input,axis=None,name=None,dimension=None)   0 for
# Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1,32768],dtype=tf.float32)        # next Q value for 4 actions
loss = tf.reduce_sum(tf.square(nextQ - Qout))       # summation of squared loss
trainer = tf.train.GradientDescentOptimizer(learning_rate=1)  # set optimization method
updateModel = trainer.minimize(loss)        # set objective function

y=0.99  # discount rate
#
exploration_rate_const=0.05
action_map={0:[-1,-1,-1],
            1:[-1,-1,0],
            2: [-1, -1, 1],
            3: [-1, 0, -1],
            4: [-1, 0, 0],
            5: [-1, 0, 1],
            6: [-1, 1, -1],
            7: [-1, 1, 0],
            8: [-1, 1, 1],
            9:[0,-1,-1],
            10:[0,-1,0],
            11: [0, -1, 1],
            12: [0, 0, -1],
            13: [0, 0, 0],
            14: [0, 0, 1],
            15: [0, 1, -1],
            16: [0, 1, 0],
            17: [0, 1, 1],
            18:[1,-1,-1],
            19:[1,-1,0],
            20: [1, -1, 1],
            21: [1, 0, -1],
            22: [1, 0, 0],
            23: [1, 0, 1],
            24: [1, 1, -1],
            25: [1, 1, 0],
            26: [1, 1, 1]
            }
action_map_flex_dir = {}
counter = 0
# action mapping
# player 1
for i1 in range(2):  # move or not
    for k1 in range(2):  # x axis
        for l1 in range(2): # y axis:
            # player 2
            for i2 in range(2):  # move or not
                for k2 in range(2):  # x axis
                    for l2 in range(2):  # y axis:
                        # player 3
                        for i3 in range(2):  # move or not
                            for k3 in range(2):  # x axis
                                for l3 in range(2):  # y axis:
                                    # player 4
                                    for i4 in range(2):  # move or not
                                        for k4 in range(2):  # x axis
                                            for l4 in range(2):  # y axis:
                                                # player 5
                                                for i5 in range(2):  # move or not
                                                    for k5 in range(2):  # x axis
                                                        for l5 in range(2):  # y axis:
                                                            action_map_flex_dir[counter] = [i1, set_move(k1), set_move(l1),
                                                                                            i2, set_move(k2), set_move(l2),
                                                                                            i3, set_move(k3), set_move(l3),
                                                                                            i4, set_move(k4), set_move(l4),
                                                                                            i5, set_move(k5), set_move(l5)]
                                                            counter += 1
#
with tf.Session() as sess:
    # initialize weights
    sess.run(tf.global_variables_initializer())
    print("variable initalized, env reset, start first trial")
    prev_ = 0
    iter=0
    for _ in range(15000000):
      env.render()
      obs=[]
      for k1 in env.state:
        for k2 in k1:
            obs.append(k2)
      #print("start calculate Q values: ")
      obs=np.reshape(obs,(-1,187))
      action_index, allQ = sess.run([predict, Qout], feed_dict={inputs1:obs})  # var:new_data for feeding
      #print("Q values calculated, mapping action")
      action = action_map_flex_dir[action_index[0]]
      if np.random.rand(1) < exploration_rate_const*(1-(np.log10(_+1)/7.17)):  # random action
          action = env.action_space.sample()
      #print("action: ", action)
      new_obs=[]
      new_obs_raw, reward, done, info = env.step(np.array(action))
      for k1 in new_obs_raw:
        for k2 in k1:
            new_obs.append(k2)
      new_obs = np.reshape(new_obs, (-1, 187))
      # Obtain the Q' values by feeding the new state through our network
      Q_for_next_step = sess.run(Qout, feed_dict={inputs1:new_obs})
      # O btain maxQ' and set our target value for chosen action.
      max_Q_for_next_step = np.max(Q_for_next_step)
      targetQ = allQ
      targetQ[0, action_index[0]] = reward + y * max_Q_for_next_step
      # Train our network using target and predicted Q values
      sess.run(updateModel, feed_dict={inputs1: obs , nextQ: targetQ})
      obs = new_obs
      # results
      if done:
          # report
          print("iter: ",iter,", ", obs[-1][-1]*100,"% BOSS remaining HP"," steps: ",_-prev_)
          obs=env.reset()
          iter+=1
          prev_=_
    # env.close()

