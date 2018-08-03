import numpy as np 
import pandas as pd 
epision = 0.5
Learning_rate =  0.01
GAMMA = 0.9
MAX_epoch = 1
ACTIONS_N = 64
STATION_N = 3952
ACTIONS = [i for i in range(ACTIONS_N)]
def load_file():
 	Result = pd.read_table('result12221.txt',sep=' ', header=None,dtype=float, na_filter=False)
 	return Result 

def Action(S,R):
	state_actions = R.iloc[S, :]
	if (np.random.uniform() > epision ):
		A = np.random.choice(ACTIONS)
	else:   # act greedy
		A = state_actions.idxmax(ACTIONS_N)    
	return A

def Reward(S,A,R):
	r = np.mean(R.values[S]) 
	if R.values[S,A] > r:
		rew = 0.1
	else:
		if R.values[S,A] == r:
			rew = 0
		else :
			rew = -0.1
	return rew 
def Optimizer1(R):
	
	for i in range(MAX_epoch):
		S = 0
		print(i)
		
		while S < STATION_N-1:
			S_ = S + 1
			A = Action(S,R)
			rew = Reward(S,A,R)
			q_predict = R.values[S,A]
			q_target = rew + GAMMA*R.values[S_,:].max()
			# if R.values[S,A]<5:
			R.values[S,A] += Learning_rate * (q_target - q_predict)
			S += 1
		if S == STATION_N-1:
			A = Action(S,R)
			rew = Reward(S,A,R)
			q_predict = R.values[S,A]
			q_target = rew + GAMMA*R.values[0,:].max()
			R.values[S,A] += Learning_rate * (q_target - q_predict)
	return R

# if __name__ == '__main__':
# 	R = load_file() 
# 	R_ = Optimizer(R)
# 	R_.to_csv('11.txt')
# 	exit()
# 	R_ = np.array(R_)

# 	result = open('11.txt','w')
# 	np.savetxt(result,R_ ,fmt="%s")

	