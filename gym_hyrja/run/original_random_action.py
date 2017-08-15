import gym
import gym_hyrja

env = gym.make("Hyrja-v0")
observation = env.reset()
for _ in range(15000):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)
  if done:
    env.reset()

env.close()

