import gym
from gym import envs
import gym_Jass.envs.Jass_env

env = gym.make('Jass-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        print(observation)
        action = env.action_space.sample()
        print(env._decode_action(action))
        print(env.action_space)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()