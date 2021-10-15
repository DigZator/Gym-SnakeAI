# Snake game AI

#### Created in the [Gym-SnakeAI](https://github.com/grantsrb/Gym-Snake) Environment

## Description

This is an attempt to solve the Snake Game using Tabular Reinforcement Learning Methods, namely SARSA (λ).

![wuxing_rel](https://user-images.githubusercontent.com/87714053/136929595-33cb2e1d-00c4-4f33-acf8-c5407f3a7919.gif)

This repository has 2 methods for solving the game -

### 4 Food detector and 3 Block detectors

In this method a state is defined by the relative direction towards the fruit and whether the relative adjacent grid cells are Snake body or out of the grid.
- Relative Direction of Food 
    - Dividing the relative grid into 4 sectors by lines which are equally inclined to the axis
    - With respect to the Snake Head, the Food lies in one of these sectors
    - It leads the agent towards the Goal/Positive Reward
- Adjacent Grid
    - It tells whether the grid cells that are adjacent to the Snake Head are either Snake Body or out of bounds or a simple grid cell
    - It is represented by a number between 0 and 7
    - This number corresponds to its binary representation 000, 001, 010,..., 110, 111 - In the binary representation the least significant digit signifies whether the grid cell to the Right of the head is Snake body or out of bounds or not, and similarly the most significant digit tells us about the grid cell to the Left of the head, and the 2nd least significant digit corresponds to the grid cell in front of the head
    - It tells the agent where it might receive a negative reward

These state parameters set the state space to 4x8 = 24 states
This method can be found in the wuxing.py file, the algorithm takes time to develop a policy, and sometimes the policy isn't optimal, so there is also an Optimal Policy predefined which can be used instead, by commenting out the line which calls on the wuxing() function.

#### Reward Earned over multiple runs
![Reward](https://user-images.githubusercontent.com/87714053/137489078-ac52670b-fdc8-442e-9919-8d74d47c18f1.png)
![Reward and Running Avg](https://user-images.githubusercontent.com/87714053/136941082-82846f97-09c0-47fd-bc03-8165d2b6f76c.png)

 
### VI and QL

In this method once the game starts or the Food is consumed, the agent performs Value Iteration on the grid to propagate the sparse reward and get a policy based on that. This policy may or may not be optimal so the agent performs Q-Learning on Policy by creating a copy of the environment, and when an optimal policy is achieve then enacting it on the original environment.

This method was not completed because the first method gave better results.

## Execution

The program can be executed by running the wuxing.py file - to run the learning algorithm and test it once.
To get graphs regarding some property, one can make changes to wuxing_rel_graphs.py.
To see how the agent behaves when multiple snakes are present on the grid simultaneously.
To generate walls in the grid space, edit the env files.

![Multi](https://user-images.githubusercontent.com/87714053/136929050-b6534e87-9bb2-4bb7-9d84-4c14a4a6bec6.gif)

![wuxing_rel - bet_wall](https://user-images.githubusercontent.com/87714053/136930849-9e458042-e1f4-45bf-8e4d-55bfe8c6e89e.gif)

# Environment used

## gym-snake

#### Created in response to OpenAI's [Requests for Research 2.0](https://blog.openai.com/requests-for-research-2/)

## Description
gym-snake is a multi-agent implementation of the classic game [snake](https://www.youtube.com/watch?v=wDbTP0B94AM) that is made as an OpenAI gym environment.

The two environments this repo offers are snake-v0 and snake-plural-v0. snake-v0 is the classic snake game. See the section on SnakeEnv for more details. snake-plural-v0 is a version of snake with multiple snakes and multiple snake foods on the map. See the section on SnakeExtraHardEnv for more details. 

Many of the aspects of the game can be changed for both environments. See the Game Details section for specifics.

## Dependencies
- pip
- gym
- numpy
- matplotlib

## Installation
1. Clone this repository
2. Navigate to the cloned repository
3. Run command `$ pip install -e ./`
