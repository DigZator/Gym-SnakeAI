import numpy as np
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
snake = snakes_array[0]

#print(snake.head,snake.direction)
#print(observation.shape)
#print(observation)

nx,ny,nc = observation.shape

#Returns a general direction to the Food
def detector(hloc,obs):
	dire = 0
	floc = [0,0]
	nx,ny,nc = obs.shape
	#Declaring and finding the Food Location
	for x in range(0,nx,10):
		for y in range(0,ny,10):
			if (np.array_equal(obs[x][y],FOOD_COLOR)):
				floc = [y//10,x//10]
				x = nx - 1 
				y = ny - 1
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
	#[LCR] - LEFT, CENTER, RIGHT - Reletive to the direction the snake is facing in - Since it will be binary, we can assign numbers from 0 to 7
	outbin = 0
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

def better_det(hloc,obs):
	dire = [0,0,0,0] #URDL
	floc = [0,0]
	nx,ny,nc = obs.shape
	#Declaring and finding the Food Location
	for x in range(0,nx,10):
		for y in range(0,ny,10):
			if (np.array_equal(obs[x][y],FOOD_COLOR)):
				floc = [y//10,x//10]
				x = nx - 1
				y = ny - 1
	#dire = [UP,RIGHT,DOWN,LEFT]
	hx, hy = hloc[0],hloc[1]
	fx, fy = floc[0],floc[1]
	diffx = fx-hx
	diffy = fy-hy
	if diffx > 0:
		dire[1] = 1
	elif diffx < 0:
		dire[3] = 1
	if diffy > 0:
		dire[2] = 1
	elif diffy < 0:
		dire[0] = 1
	key = ''.join(map(str,dire))
	Pot = {"1000": 0,
		   "0100": 2,
		   "0010": 4,
		   "0001": 6,
		   "1100": 1,
		   "0110": 3,
		   "0011": 5,
		   "1001": 7}
	return Pot[key]

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

# hdir\bet_det  1000 0100 0010 0001 1100 0110 0011 1001
#   0           1000 0100 0010 0001 1100 0110 0011 1001
#   1           0001 1000 0100 0010 1001 1100 0110 0011
#   2           0010 0001 1000 0100 0011 1001 1100 0110
#   3           0100 0010 0001 1000 0110 0011 1001 1100
def bet_rel_det(hdir,det):
	rel = [[0,1,2,3,4,5,6,7],
		   [2,3,4,5,6,7,0,1],
		   [4,5,6,7,0,1,2,3],
		   [6,7,0,1,2,3,4,5]]
	return rel[hdir][det]

def wuxing(env,n_episode = 1000,gamma = 0.9,α = 0.5,lmbd = 0.9):
	obs = env.reset()
	epn = 0
	#Initializing the Policy
	Pol = {det : {bod : 1 for bod in range(8)} for det in range(4)}
	#Pol = {0 :{0: 1, 1: 1, 2: 0, 3: 0, 4: 1, 5: 1, 6: 2, 7: 2},
	#   	   1 :{0: 2, 1: 1, 2: 2, 3: 0, 4: 2, 5: 1, 6: 2, 7: 2},
	#   	   2 :{0: 2, 1: 0, 2: 2, 3: 0, 4: 2, 5: 1, 6: 2, 7: 0},
	#   	   3 :{0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 2, 7: 2}}
	#Initalizing Q-values
	Q = {det : {bod : {act : 0 for act in range(3)} for bod in range(8)} for det in range(4)}
	#Episodes
	while (epn < n_episode):
		print(epn)
		obs = env.reset()
		#env.render()
		end = False
		count = 0
		#hloc = snake.head
		#hdir = snake.direction
		food = 0
		
		#Eligibility Traces
		E = {det : {bod : {act : 0 for act in range(3)} for bod in range(8)} for det in range(4)}

		#Running an episode
		while (not end):
			env.render()
			ε = (n_episode/10)/((n_episode/10)+epn)
			#if epn < n_episode//20:
			#	ε = 0.9
			
			# Controller
			game_controller = env.controller
			
			# Grid
			grid_object = game_controller.grid
			grid_pixels = grid_object.grid
			
			# Snake(s)
			snakes_array = game_controller.snakes
			snake = snakes_array[0]
			hloc = snake.head
			hdir = snake.direction
			#print(hloc,hdir,"Old")
			
			#Declaring the blocked directions and direction of the Food
			bod = boder(snake.head[0],snake.head[1],snake.direction,obs)
			det = detector(hloc,obs)
			rdet = rel_det(hdir,det)
			#print(bod,det,rdet,"Old")

			#ε - Greedy Action Selector
			A = Pol[rdet][bod] if (np.random.random_sample() > (ε)) else np.random.randint(3)

			#Taking a step
			obs, reward, end, info = env.step(rel_act(hdir,A))
			#reward = -0.1 if (reward == 0) else reward

			#New State
			nloc = snake.head
			ndir = snake.direction
			nbod = boder(snake.head[0],snake.head[1],snake.direction,obs)
			ndet = detector(nloc,obs)
			nrdet = rel_det(hdir,ndet)
			#print(nbod,ndet,nrdet,"New")

			#Since the env requires an extra step to end the episode
			if (reward == -1):
				obs, _, end, info = env.step(rel_act(ndir,A))
			if reward == 1:
				food = food + 1

			#Target
			targe = reward + (gamma*Q[nrdet][nbod][Pol[nrdet][nbod]]) - Q[rdet][bod][A]
			#print(reward)
			
			#Spiking the Eligibility Traces
			E[rdet][bod][A] = E[rdet][bod][A] + 1

			#Sweeping through the states to reduce Eligibility and update Q-Value according 
			for sdet in Q:
				for sbod in Q[sdet]:
					max_a = Pol[sdet][sbod]
					for sa in range(3):
						Q[sdet][sbod][sa] = Q[sdet][sbod][sa] + (α*targe*E[sdet][sbod][sa])
						E[sdet][sbod][sa] = gamma*lmbd*E[sdet][sbod][sa]
						max_a = sa if Q[sdet][sbod][sa] > Q[sdet][sbod][max_a] else max_a
					Pol[sdet][sbod] = max_a
			#else:
			#	obs,reward,end,info = env.step(A)
			#	print(reward,end)
		epn = epn + 1
		print(food)
	print(Q)
	return Pol

def wuxing_bet(env,n_episode = 1000,gamma = 0.9,α = 0.5,lmbd = 0.9):
	obs = env.reset()
	epn = 0
	#Initializing the Policy
	Pol = {det : {bod : 1 for bod in range(8)} for det in range(8)}

	#Initalizing Q-values
	Q = {det : {bod : {act : 0 for act in range(3)} for bod in range(8)} for det in range(8)}

	#Episodes
	while (epn < n_episode):
		food = 0
		print(epn)
		obs = env.reset()
		#env.render()
		end = False
		#Eligibility Traces
		E = {det : {bod : {act : 0 for act in range(3)} for bod in range(8)} for det in range(8)}

		#Running an episode
		while (not end):
			#if (epn == 1 or epn == 10 or epn == 100 or epn == 500 or epn == 1000 or epn == 2500):
			#	env.render()
			
			ε = (n_episode/10)/((n_episode/10)+epn)

			# Controller
			game_controller = env.controller

			# Grid
			grid_object = game_controller.grid
			grid_pixels = grid_object.grid

			# Snake(s)
			snakes_array = game_controller.snakes
			snake = snakes_array[0]
			hloc = snake.head
			hdir = snake.direction

			#Declaring the blocked directions and direction of the Food
			bod = boder(snake.head[0],snake.head[1],snake.direction,obs)
			det = better_det(hloc,obs)
			rdet = bet_rel_det(hdir,det)
			#print("Old",hloc,hdir,bod,det,rdet)

			#ε - Greedy Action Selector
			A = Pol[rdet][bod] if (np.random.random_sample() > ε) else np.random.randint(3)

			#Taking a step
			obs, reward, end, info = env.step(rel_act(hdir,A))
			reward = -0.1 if (reward == 0) else reward

			#New State
			nloc = snake.head
			ndir = snake.direction
			nbod = boder(snake.head[0],snake.head[1],snake.direction,obs)
			ndet = better_det(nloc,obs)
			nrdet = bet_rel_det(hdir,ndet)
			#print("New",nloc,ndir,nbod,ndet,nrdet)

			#Since the env requires an extra step to end the episode
			if (reward == -1):
				obs, _, end, info = env.step(rel_act(ndir,A))
			if reward == 1:
				food = food + 1

			#Target
			targe = reward + (gamma*Q[nrdet][nbod][Pol[nrdet][nbod]]) - Q[rdet][bod][A]

			#Spiking the Eligibility Traces
			E[rdet][bod][A] = E[rdet][bod][A] + 1

			#Sweeping through the states to reduce Eligibility and update Q-Value according 
			for sdet in Q:
				for sbod in Q[sdet]:
					max_a = Pol[sdet][sbod]
					for sa in range(3):
						Q[sdet][sbod][sa] = Q[sdet][sbod][sa] + (α*targe*E[sdet][sbod][sa])
						E[sdet][sbod][sa] = gamma*lmbd*E[sdet][sbod][sa]
						max_a = sa if Q[sdet][sbod][sa] > Q[sdet][sbod][max_a] else max_a
					Pol[sdet][sbod] = max_a
			#else:
			#	obs,reward,end,info = env.step(A)
			#	print(reward,end)
		epn = epn + 1
		print(food)
	#print(Q)
	return Pol

#Pol = {0 :{0: 1, 1: 1, 2: 2, 3: 0, 4: 1, 5: 1, 6: 2, 7: 2},
#	   1 :{0: 2, 1: 1, 2: 2, 3: 0, 4: 2, 5: 1, 6: 2, 7: 2},
#	   2 :{0: 2, 1: 0, 2: 0, 3: 0, 4: 2, 5: 1, 6: 2, 7: 0},
#	   3 :{0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 2, 7: 2}}

#Pol = wuxing(env,5000,0.5,0.7,0.5)
##Pol = wuxing(env,5000,0.1,0.8,0.9)
#for det in Pol:
#	print(det,Pol[det])
##print(Pol)
#env.close()

def test_pol(Pol,SIZE = [8,8],rep_step = False):
	envt = gym.make('snake-v0')
	envt.grid_size = SIZE
	envt.unit_size = 10
	envt.unit_gap = 1
	envt.snake_size = 3
	envt.n_snakes = 1
	envt.n_foods = 1

	end = False
	ss = 0
	obs = envt.reset()
	nx,ny,nc = obs.shape
	while (not end):
		envt.render()
		# Controller
		game_controller = envt.controller
		
		# Grid
		grid_object = game_controller.grid
		grid_pixels = grid_object.grid
		
		# Snake(s)
		snakes_array = game_controller.snakes
		snake = snakes_array[0]
	
		hloc = snake.head
		hdir = snake.direction
		floc = [0,0]
		for x in range(0,nx,10):
			for y in range(0,ny,10):
				if (np.array_equal(obs[x][y],FOOD_COLOR)):
					floc = [y//10,x//10]
		bod = boder(hloc[0],hloc[1],hdir,obs)
		det = detector(hloc,obs)
		rdet = rel_det(hdir,det)
		
		if rep_step:
			print("Head :",hloc,"Food :",floc,"Dire :",hdir,"Det :",det,"RDet :",rdet,"Bod :",bod)
		
		A = rel_act(hdir,Pol[rdet][bod])
		#print(A)
		obs, reward, end, info = envt.step(A)
	
		ss = ss + 1 if reward == 1 else ss
		#Since the env requires an extra step to end the episode
		if (reward == -1):
			obs, _, end, info = envt.step(A)
	return ss

#print("Score - ",test_pol(Pol,[15,15]))

def test_pol_bet(Pol,SIZE = [8,8],rep_step = False):
	envt = gym.make('snake-v0')
	envt.grid_size = SIZE
	envt.unit_size = 10
	envt.unit_gap = 1
	envt.snake_size = 3
	envt.n_snakes = 1
	envt.n_foods = 1

	end = False
	ss = 0
	obs = envt.reset()
	nx,ny,nc = obs.shape
	while (not end):
		envt.render()
		# Controller
		game_controller = envt.controller
		
		# Grid
		grid_object = game_controller.grid
		grid_pixels = grid_object.grid
		
		# Snake(s)
		snakes_array = game_controller.snakes
		snake = snakes_array[0]
	
		hloc = snake.head
		hdir = snake.direction
		floc = [0,0]
		for x in range(0,nx,10):
			for y in range(0,ny,10):
				if (np.array_equal(obs[x][y],FOOD_COLOR)):
					floc = [y//10,x//10]
		bod = boder(hloc[0],hloc[1],hdir,obs)
		det = better_det(hloc,obs)
		rdet = bet_rel_det(hdir,det)
		
		if rep_step:
			print("Head :",hloc,"Food :",floc,"Dire :",hdir,"Det :",det,"RDet :",rdet,"Bod :",bod)
		
		A = rel_act(hdir,Pol[rdet][bod])
		#print(A)
		obs, reward, end, info = envt.step(A)
	
		ss = ss + 1 if reward == 1 else ss
		#Since the env requires an extra step to end the episode
		if (reward == -1):
			obs, _, end, info = envt.step(A)
	return ss

Pol = wuxing_bet(env,5000)
for det in Pol:
	print(det, Pol[det])
print("Score - ",test_pol_bet(Pol,[8,8]))

#for _ in range(20):
#	Pol = wuxing_rel(env,5000,0.5,0.7,0.5)
#	for det in Pol:
#		print(det,Pol[det])
#env.grid_size = [12,8]
#for det in Pol:
#	print(det,Pol[det])
#print("Snake Size : ",test_pol(Pol,[20,20],True)))