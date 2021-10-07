import numpy as np
import matplotlib
import gym
import gym_snake

#Making the environment
env = gym.make('snake-v0')
env.grid_size = [8,8]
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
print(observation.shape)
#print(observation)

nx,ny,nc = observation.shape

#Returns a general direction to the Food
def detector(hloc,floc):
	dire = 0
	#dire = [UP,RIGHT,DOWN,LEFT]
	hx, hy = hloc[0],hloc[1]
	fx, fy = floc[0],floc[1]
	diffx,diffy = (fx-hx),(fy-hy)
	if ((diffx == 0) and (diffy == 0)):
		dire = 0
	#elif ((abs(diffx) - abs(diffy)) > 0): 	#The axis where the abosolute difference is greater should have the direction 
	#	if (diffx > 0):						#If depending the different it would be along the axis in a positive direction
	#		dire = 1
	#	elif (diffx < 0):					#or negative diretion
	#		dire = 3
	#elif ((abs(diffx) - abs(diffy)) < 0):
	#	if (diffy > 0):
	#		dire = 2#0
	#	elif (diffy < 0):
	#		dire = 0#2
	if ((abs(diffx) == abs(diffy)) and (diffx != 0)): #If the absolute difference is same then it lies on the dividing lines
		if (diffx > 0):									#So the direction assigned is the next sector which is in clockwise direction
			if (diffy > 0):
				dire = 2
			else:
				dire = 1
		else:
			if (diffy > 0):
				dire = 3
			else:
				dire = 0
	elif (diffx > diffy):
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

print(detector([0,0],[-5,1]))
#np.array_equal(obs[L[0]][L[1]], BODY_COLOR)

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


def wuxing(env,n_episode = 1000,gamma = 0.9,α = 0.5,lmbd = 0.9):
	obs = env.reset()
	epn = 0
	#Initializing the Policy
	Pol = {d : {det : {bod : np.random.randint(3) for bod in range(8)} for det in range(4)} for d in range(4)}
	#Episodes
	while (epn < n_episode):
		print(epn)
		obs = env.reset()
		#env.render()
		end = False
		#hloc = snake.head
		#hdir = snake.direction

		#Initalizing Q-values
		Q = {d : {det : {bod : {act : 0 for act in range(3)} for bod in range(8)} for det in range(4)} for d in range(4)}
		#Pol = {d : {det : {bod : 0 for bod in range(8)} for det in range(4)} for d in range(4)}
		
		#Running an episode
		while (not end):
			#Eligibility Traces
			E = {d : {det : {bod : {act : 0 for act in range(3)} for bod in range(8)} for det in range(4)} for d in range(4)}
			
			#Random Action Selector
			A = np.random.randint(3)
			#env.render()
			ε = 450/(epn+450)
			
			# Controller
			game_controller = env.controller
			
			# Grid
			grid_object = game_controller.grid
			grid_pixels = grid_object.grid
			
			# Snake(s)
			snakes_array = game_controller.snakes
			snake = snakes_array[0]
			#if hasattr(snake,'head'):
			hloc = snake.head
			hdir = snake.direction
			#print(hloc,hdir,"Old")

			#Declaring and finding the Food Location
			floc = [0,0]
			for x in range(0,nx,10):
				for y in range(0,ny,10):
					if (np.array_equal(obs[x][y],FOOD_COLOR)):
						floc = [y/10,x/10]
			
			#Declaring the blocked directions and direction of the Food
			bod = boder(snake.head[0],snake.head[1],hdir,obs)
			det = detector(hloc,floc)

			#ε - Greedy Action Selector
			A = Pol[hdir][det][bod] if (np.random.random_sample() > (ε)) else A

			#Taking a step
			obs, reward, end, info = env.step(rel_act(hdir,A))
			#print(reward,end)

			#New State
			for x in range(0,nx,10):
				for y in range(0,ny,10):
					if (np.array_equal(obs[x][y],FOOD_COLOR)):
						floc = [x/10,y/10]
			nloc = snake.head
			ndir = snake.direction
			nbod = boder(snake.head[0],snake.head[1],snake.direction,obs)
			ndet = detector(nloc,floc)
			#print(nloc,ndir)

			#Since the env requires an extra step to end the episode
			if (reward == -1):
				obs, _, end, info = env.step(rel_act(ndir,A))

			#Target
			targe = reward + (gamma*Q[ndir][ndet][nbod][Pol[ndir][ndet][nbod]]) - Q[hdir][det][bod][A]
			
			#Spiking the Eligibility Traces
			E[hdir][det][bod][A] = E[hdir][det][bod][A] + 1

			#Sweeping through the states to reduce Eligibility and update Q-Value according 
			for sd in Q:
				for sdet in Q[sd]:
					for sbod in Q[sd][sdet]:
						max_a = Pol[sd][sdet][sbod]
						for sa in range(3):
							Q[sd][sdet][sbod][sa] = Q[sd][sdet][sbod][sa] + (α*targe*E[sd][sdet][sbod][sa])
							E[sd][sdet][sbod][sa] = gamma*lmbd*E[sd][sdet][sbod][sa]
							max_a = sa if Q[sd][sdet][sbod][sa] > Q[sd][sdet][sbod][max_a] else max_a
						Pol[sd][sdet][sbod] = max_a
			#else:
			#	obs,reward,end,info = env.step(A)
			#	print(reward,end)
		epn = epn + 1
	return Pol

def wuxing_rel(env,n_episode = 1000,gamma = 0.9,α = 0.5,lmbd = 0.9):
	obs = env.reset()
	epn = 0
	#Initializing the Policy
	Pol = {det : {bod : 0 for bod in range(8)} for det in range(4)}
	#Pol = {0 :{0: 1, 1: 1, 2: 0, 3: 0, 4: 1, 5: 1, 6: 2, 7: 2},
	#   	   1 :{0: 2, 1: 1, 2: 2, 3: 0, 4: 2, 5: 1, 6: 2, 7: 2},
	#   	   2 :{0: 2, 1: 0, 2: 2, 3: 0, 4: 2, 5: 1, 6: 2, 7: 0},
	#   	   3 :{0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 2, 7: 2}}
	#Episodes
	while (epn < n_episode):
		print(epn)
		obs = env.reset()
		#env.render()
		end = False
		#hloc = snake.head
		#hdir = snake.direction

		#Initalizing Q-values
		Q = {det : {bod : {act : 0 for act in range(3)} for bod in range(8)} for det in range(4)}
		#Pol = {d : {det : {bod : 0 for bod in range(8)} for det in range(4)} for d in range(4)}
		
		#Running an episode
		while (not end):
			#env.render()
			#Eligibility Traces
			E = {det : {bod : {act : 0 for act in range(3)} for bod in range(8)} for det in range(4)}
			
			#Random Action Selector
			A = np.random.randint(3)
			#env.render()
			ε = (n_episode/10)/((n_episode/10)+100)
			
			# Controller
			game_controller = env.controller
			
			# Grid
			grid_object = game_controller.grid
			grid_pixels = grid_object.grid
			
			# Snake(s)
			snakes_array = game_controller.snakes
			snake = snakes_array[0]
			#if hasattr(snake,'head'):
			hloc = snake.head
			hdir = snake.direction
			#print(hloc,hdir,"Old")

			#Declaring and finding the Food Location
			floc = [0,0]
			for x in range(0,nx,10):
				for y in range(0,ny,10):
					if (np.array_equal(obs[x][y],FOOD_COLOR)):
						floc = [y/10,x/10]
			
			#Declaring the blocked directions and direction of the Food
			bod = boder(snake.head[0],snake.head[1],snake.direction,obs)
			det = detector(hloc,floc)
			rdet = rel_det(hdir,det)

			#ε - Greedy Action Selector
			A = Pol[rdet][bod] if (np.random.random_sample() > (ε)) else A

			#Taking a step
			obs, reward, end, info = env.step(rel_act(hdir,A))
			#reward = -0.01 if (reward == 0) else reward
			#print(reward,end)

			#New State
			for x in range(0,nx,10):
				for y in range(0,ny,10):
					if (np.array_equal(obs[x][y],FOOD_COLOR)):
						floc = [x/10,y/10]
			nloc = snake.head
			ndir = snake.direction
			nbod = boder(snake.head[0],snake.head[1],snake.direction,obs)
			ndet = detector(nloc,floc)
			nrdet = rel_det(hdir,ndet)
			#print(nloc,ndir)

			#Since the env requires an extra step to end the episode
			if (reward == -1):
				obs, _, end, info = env.step(rel_act(ndir,A))
				reward = -10

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
	return Pol

#Pol = 	{0:
#			{0 : {0:1, 1:1, 2:0, 3:0, 4:1, 5:1, 6:2, 7:0}, 1 : {0:2, 1:1, 2:2, 3:0, 4:2, 5:1, 6:2, 7:0}, 2 : {0:0, 1:0, 2:0, 3:0, 4:2, 5:1, 6:2, 7:0}, 3 : {0:0, 1:0, 2:0, 3:0, 4:1, 5:1, 6:2, 7:0}},
#		 1:
#		 	{0 : {0:1, 1:1, 2:0, 3:0, 4:1, 5:1, 6:2, 7:0}, 1 : {0:2, 1:1, 2:2, 3:0, 4:2, 5:1, 6:2, 7:0}, 2 : {0:0, 1:0, 2:0, 3:0, 4:2, 5:1, 6:2, 7:0}, 3 : {0:0, 1:0, 2:0, 3:0, 4:1, 5:1, 6:2, 7:0}},
#		 2:
#		 	{0 : {0:1, 1:1, 2:0, 3:0, 4:1, 5:1, 6:2, 7:0}, 1 : {0:2, 1:1, 2:2, 3:0, 4:2, 5:1, 6:2, 7:0}, 2 : {0:0, 1:0, 2:0, 3:0, 4:2, 5:1, 6:2, 7:0}, 3 : {0:0, 1:0, 2:0, 3:0, 4:1, 5:1, 6:2, 7:0}},
#		 3:
#		 	{0 : {0:1, 1:1, 2:0, 3:0, 4:1, 5:1, 6:2, 7:0}, 1 : {0:2, 1:1, 2:2, 3:0, 4:2, 5:1, 6:2, 7:0}, 2 : {0:0, 1:0, 2:0, 3:0, 4:2, 5:1, 6:2, 7:0}, 3 : {0:0, 1:0, 2:0, 3:0, 4:1, 5:1, 6:2, 7:0}}}


#Pol = wuxing(env,1000)
##print(Pol)
#for d in Pol:
#	for det in Pol[d]:
#		print(d,det,Pol[d][det])
#obs = env.reset()
#end = False
#
#while (not end):
#	env.render()
#	hloc = snake.head
#	hdir = snake.direction
#	floc = [0,0]
#	for x in range(0,nx,10):
#		for y in range(0,ny,10):
#			if (np.array_equal(obs[x][y],FOOD_COLOR)):
#				floc = [x,y]
#	bod = boder(hloc[0],hloc[1],hdir,obs)
#	det = detector(hloc,floc)
#	A = Pol[hdir][det][bod]
#	obs, reward, end, info = env.step(rel_act(hdir,A))
#env.close()

Pol = {0 :{0: 1, 1: 1, 2: 0, 3: 0, 4: 1, 5: 1, 6: 2, 7: 2},
	   1 :{0: 2, 1: 1, 2: 2, 3: 0, 4: 2, 5: 1, 6: 2, 7: 2},
	   2 :{0: 2, 1: 0, 2: 2, 3: 0, 4: 2, 5: 1, 6: 2, 7: 0},
	   3 :{0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 2, 7: 2}}

Pol = wuxing_rel(env,5000,0.9)
#print(Pol)

def test_pol(Pol,SIZE = [20,20],rep_step = False):
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
					floc = [y/10,x/10]
		bod = boder(hloc[0],hloc[1],hdir,obs)
		det = detector(hloc,floc)
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

for det in Pol:
	print(det,Pol[det])
#env.grid_size = [12,8]

print("Snake Size : ",test_pol(Pol,[8,8],True))