import numpy as np
import matplotlib
import gym
import gym_snake

#Making the environment
env = gym.make('snake-v0')
env.grid_size = [15,16]
env.unit_size = 10
env.unit_gap = 1
env.snake_size = 3
env.n_snakes = 1
env.n_foods = 1

observation = env.reset() # Constructs an instance of the game


FOOD_COLOR = np.array([0, 0, 255], dtype=np.uint8)
HEAD_COLOR = np.array([255, 10 * 0, 0], dtype=np.uint8)
# Controller
game_controller = env.controller

# Grid
grid_object = game_controller.grid
grid_pixels = grid_object.grid

# Snake(s)
snakes_array = game_controller.snakes
snake = snakes_array[0]

#for _ in range(1000):
#    env.render()
#    obs, reward, done, info = env.step(env.action_space.sample()) # take a random action
#    # BODY_COLOR = np.array([1, 0, 0], dtype=np.uint8)
#    # HEAD_COLOR = np.array([255, 10 * i, 0], dtype=np.uint8)
#    # SPACE_COLOR = np.array([0, 255, 0], dtype=np.uint8)
#    # FOOD_COLOR = np.array([0, 0, 255], dtype=np.uint8)
#    print(obs.size,obs.shape)
#    for x in range(0,150,10):
#        for y in range(0,150,10):
#            if (np.array_equal(obs[x][y],np.array([0,0,255]))):
#                print(x,y)
#    # Controller
#    game_controller = env.controller
#    
#    # Grid
#    grid_object = game_controller.grid
#    grid_pixels = grid_object.grid
#    
#    # Snake(s)
#    snakes_array = game_controller.snakes
#    snake = snakes_array[0]
#    print(done)
#    if done:
#        env.reset()
#    else:
#        print(snake.direction)
def move(action,x,y,d,R,ny,nx):
    #UP = 0
    #RIGHT = 1
    #DOWN = 2
    #LEFT = 3
    end = False
    d = action
    if d == 4:
        if (x-1<0):
            end = True
        else:
            x = x-1
    elif d == 2:
        if (y+1==ny):
            end = True
        else:
            y = y+1
    elif d == 1:
        if (x+1==nx):
            end = True
        else:
            x = x+1
    elif d == 0:
        if (y-1<0):
            end = True
        else:
            y = y-1
    Rew = -10 if (end) else R[x][y][d]
    return x,y,d,Rew
def VI_QL(obs,temp_env):
    nx,ny,nz = obs.shape

    temp_control = temp_env.controller
    temp_grid = temp_control.grid
    snakes_array = temp_control.snakes
    snake = snakes_array[0]

    #Intialize Q-value, Reward and Policy
    Q = np.zeros((nx,ny,4))
    R = np.zeros((nx,ny,4))
    Pol = np.zeros((nx,ny,4))
    Fx,Fy = (0,0)

    #Finding the Food and assigning its Reward to be 1
    for x in range(0,nx,10):
        for y in range(0,ny,10):
            for d in range(4):
                #print(x,y)
                R[x][y][d] = -1
                if (np.array_equal(obs[x][y],FOOD_COLOR)):
                    R[x][y][d] = 10
                    Fx,Fy = x,y

    #def move(action,x,y,d,R):
    #    #UP = 0
    #    #RIGHT = 1
    #    #DOWN = 2
    #    #LEFT = 3
    #    end = False
    #    d = action
    #    if d == 0:
    #        if (x-1<0):
    #            end = True
    #        else:
    #            x = x-1
    #    elif d == 1:
    #        if (y+1==ny):
    #            end = True
    #        else:
    #            y = y+1
    #    elif d == 2:
    #        if (x+1==nx):
    #            end = True
    #        else:
    #            x = x+1
    #    elif d == 3:
    #        if (y-1<0):
    #            end = True
    #        else:
    #            y = y-1
    #    Rew = -10 if (end) else R[x][y][d]
    #    return x,y,d,Rew
    gamma = 0.9
    emptyQ = np.copy(R)
    prev_value_function = np.copy(emptyQ)
    end = False
    while (not end):
        for x in range(0,nx,10):
            for y in range(0,ny,10):
                for d in range(4):
                    max_a_val = -100
                    points = 0
                    max_a = 0
                    for a in range(4):
                        newx,newy,newd,Rew = move(a,x,y,d,R,ny,nx)
                        #print(prev_value_function[newx][newy][newd])
                        points = Rew + gamma*prev_value_function[newx][newy][newd]
                        if Rew == 10:
                            print(x,y)
                        #print(x,y,d,a,points,Rew)
                        if points >= max_a_val:
                            max_a_val = points
                            max_a = a
                            #print(max_a,max_a_val)
                    Q[x][y][d] = max_a_val
                    Pol[x][y][d] = max_a
                    #print(Q)
        check = 0
        print(Q[Fx][Fy+1],Pol[Fx][Fy+1])
        for x in range(0,nx,10):
            for y in range(0,ny,10):
                for d in range(4):
                    if (abs(prev_value_function[x][y][d] - Q[x][y][d])>0.00001):
                        check = 1
                        #print(Q[x][y][d])
        if check != 1:
            end = True
        else:
            prev_value_function = np.copy(Q)
            #print(Q)
    
    env.reset()
    done = False
    while (not done):
        env.render()
        cx,cy,cd = (0,0,0)
        for x in range(0,150,10):
            for y in range(0,150,10):
                if (np.array_equal(obs[x][y],HEAD_COLOR)):
                    cx = x
                    cy = y
                    cd = snake.direction
                    print(cx,cy,cd,Pol[cx][cy][cd])
        obs, reward, done, info = env.step(int(Pol[cx][cy][cd]))
#print(observation, observation.size, observation.shape)
print(observation.shape)
#VI_QL(observation,env)
#env.close()