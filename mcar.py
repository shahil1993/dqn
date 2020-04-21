import gym
from gym import spaces
env = gym.make('MountainCar-v0')
#env = gym.make('CartPole-v0')
print(env.action_space)
#> Discrete(2)
print(env.observation_space)
#> Box(4,)

print(env.observation_space.high)

print(env.observation_space.low)

print(env.observation_space.shape[0])

space = spaces.Discrete(8) # Set with 8 elements {0, 1, 2, ..., 7}
print(space)
x = space.sample()
print(x)
assert space.contains(x)
assert space.n == 9