import gym
from gym import envs
import gym_Jass.envs.Jass_env

env = gym.make('Jass-v0')
for i_episode in range(200):
    observation = env.reset()
    for t in range(1000):
        action = int(env.action_space.sample())

        observation, reward, done, info = env.step(action)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()

