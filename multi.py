import numpy as np
import matplotlib
import gym
import gym_snake

#Making the environment
env = gym.make('snake-v0')
env.grid_size = [20,20]
env.unit_size = 10
env.unit_gap = 1
env.snake_size = 3
env.n_snakes = 2
env.n_foods = 2

observation = env.reset() # Constructs an instance of the game

nx,ny,nc = observation.shape

#Colours
FOOD_COLOR = np.array([0, 0, 255], dtype=np.uint8)
HEAD_COLOR = np.array([255, 10 * 0, 0], dtype=np.uint8)
BODY_COLOR = np.array([1, 0, 0], dtype=np.uint8)

# Controller
game_controller = env.controller

# Grid
grid_object = game_controller.grid
grid_pixels = grid_object.grid

# Snake(s)
snakes_array = game_controller.snakes
snake1 = snakes_array[0]
snake2 = snakes_array[1]

#Returns a general direction to the Food
def detector(hloc,floc):
	dire = 0
	#dire = [UP,RIGHT,DOWN,LEFT]
	hx, hy = hloc[0],hloc[1]
	fx, fy = floc[0],floc[1]
	diffx,diffy = (fx-hx),(fy-hy)
	if ((diffx == 0) and (diffy == 0)):
		dire = 0
	if ((abs(diffx) == abs(diffy)) and (diffx != 0)): #If the absolute difference is same then it lies on the dividing lines
		if (diffx > 0):								  #So the direction assigned is the next sector which is in clockwise direction
			if (diffy > 0):
				dire = 2
			else:
				dire = 1
		else:
			if (diffy > 0):
				dire = 3
			else:
				dire = 0
	elif (diffx > diffy):							  #Simple Compass which uses 2 equations (x>y and x>-y) to point towards the 4 cardinal directions
		if (diffx < -1*diffy):
			dire = 0
		elif (diffx > diffy*-1):
			dire = 1
	elif (diffx < diffy):
		if (diffx > diffy*-1):
			dire = 2
		elif (diffx < diffy*-1):
			dire = 3
	else:
		dire = 0

	return dire

#Body and Border detector
def boder(hx,hy,hd,obs):
	nx,ny,nc = obs.shape
	outbin = 0 #[LCR] - LEFT, CENTER, RIGHT - Reletive to the direction the snake is facing in - Since it will be binary, we can assign numbers from 0 to 7
	#d - UP,RIGHT,DOWN,LEFT

	if (hd == 0):
		L = ((hx-1)*10,(hy)*10)
		C = (10*hx, 10*(hy-1))
		R = (10*(hx+1),10*hy)

	elif (hd == 1):
		L = (10*hx,10*(hy-1))
		C = (10*(hx+1), 10*hy)
		R = (10*hx,10*(hy+1))

	elif (hd == 2):
		L = (10*(hx+1),10*hy)
		C = (10*hx, 10*(hy+1))
		R = (10*(hx-1),10*hy)

	elif (hd == 3):
		L = (10*hx,10*(hy+1))
		C = (10*(hx-1), 10*hy)
		R = (10*hx,10*(hy-1))

	if (L[0] < 0 or L[0] >= ny or L[1] < 0 or L[1] >= nx):
		outbin = outbin + 4
	elif (np.array_equal(obs[L[1]][L[0]], BODY_COLOR)):
		outbin = outbin + 4
	if (C[0] < 0 or C[0] >= ny or C[1] < 0 or C[1] >= nx):
		outbin = outbin + 2
	elif (np.array_equal(obs[C[1]][C[0]], BODY_COLOR)):
		outbin = outbin + 2
	if (R[0] < 0 or R[0] >= ny or R[1] < 0 or R[1] >= nx):
		outbin = outbin + 1
	elif (np.array_equal(obs[R[1]][R[0]], BODY_COLOR)):
		outbin = outbin + 1

	return outbin

def rel_act(hdir,act):
	rel = [[3,0,1],
		   [0,1,2],
		   [1,2,3],
		   [2,3,0]]
	return (rel[hdir][act])

#  hdir\det  0   1   2   3
#    0       0   1   2   3  
#    1       3   0   1   2  
#    2       2   3   0   1  
#    3       1   2   3   0  

def rel_det(hdir,det):
	rel = [[0,1,2,3],
		   [3,0,1,2],
		   [2,3,0,1],
		   [1,2,3,0]]
	return rel[hdir][det]

Pol = {0 :{0: 1, 1: 1, 2: 2, 3: 0, 4: 1, 5: 1, 6: 2, 7: 2},
	   1 :{0: 2, 1: 1, 2: 2, 3: 0, 4: 2, 5: 1, 6: 2, 7: 2},
	   2 :{0: 2, 1: 0, 2: 0, 3: 0, 4: 2, 5: 1, 6: 2, 7: 0},
	   3 :{0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 2, 7: 2}}

end = False
end2 = False

obs = env.reset()

while (not end):
	env.render()
	# Controller
	game_controller = env.controller
	
	# Grid
	grid_object = game_controller.grid
	grid_pixels = grid_object.grid
	
	# Snake(s)
	snakes_array = game_controller.snakes
	snake1 = snakes_array[0]
	snake2 = snakes_array[1]

	hloc1 = snake1.head
	hdir1 = snake1.direction
	floc = [0,0]
	floc1 = [0,0]
	check = 0
	for x in range(0,nx,10):
		for y in range(0,ny,10):
			if (np.array_equal(obs[x][y],FOOD_COLOR)):
				if check == 1:
					floc = [y/10,x/10]
				else:
					floc1 = [y/10,x/10]
				check = 1
	if ((floc[0] - hloc1[0])*(floc[0] - hloc1[0])+(floc[1] - hloc1[1])*(floc[1] - hloc1[1]) > (floc1[0] - hloc1[0])*(floc1[0] - hloc1[0])+(floc1[1] - hloc1[1])*(floc1[1] - hloc1[1])):
		temp = [0,0]
		temp[0],temp[1] = floc1[0],floc1[1]
		floc1[0],floc1[1] = floc[0],floc[1]
		floc[0],floc[1] = temp[0],temp[1]

	bod1 = boder(hloc1[0],hloc1[1],hdir1,obs)
	det1 = detector(hloc1,floc)
	rdet1 = rel_det(hdir1,det1)
	
	hloc2 = snake2.head
	hdir2 = snake2.direction
	bod2 = boder(hloc2[0],hloc2[1],hdir2,obs)
	det2 = detector(hloc2,floc1)
	rdet2 = rel_det(hdir2,det2)
	
	#if rep_step:
	#	print("Head :",hloc,"Food :",floc,"Dire :",hdir,"Det :",det,"RDet :",rdet,"Bod :",bod)
		
	A = [rel_act(hdir1,Pol[rdet1][bod1]), rel_act(hdir2,Pol[rdet2][bod2])]
	#print(A)
	obs, reward, end, info = env.step(A)
	
	#ss = ss + 1 if reward == 1 else ss
	#Since the env requires an extra step to end the episode
	if (reward == -1):
		obs, _, end, info = env.step(A)