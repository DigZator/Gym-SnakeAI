import numpy
import matplotlib
import gym
import gym_snake

#Making the environment
env = gym.make('snake-v0')
env.grid_size = [15,15]
env.unit_size = 10
env.unit_gap = 1
env.snake_size = 3
env.n_snakes = 1
env.n_foods = 1

observation = env.reset() # Constructs an instance of the game

for _ in range(1000):
    env.render()
    obs, reward, done, info = env.step(env.action_space.sample()) # take a random action
    # BODY_COLOR = np.array([1, 0, 0], dtype=np.uint8)
    # HEAD_COLOR = np.array([255, 10 * i, 0], dtype=np.uint8)
    # SPACE_COLOR = np.array([0, 255, 0], dtype=np.uint8)
    # FOOD_COLOR = np.array([0, 0, 255], dtype=np.uint8)
    if done:
        env.reset()
env.close()