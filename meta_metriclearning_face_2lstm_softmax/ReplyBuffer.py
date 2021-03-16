from collections import deque
import math, random
import torch

class ReplayBuffer(object):
	def __init__(self,ReplayMemory):
		
		#self.state_size = state_size
		self.buffer = deque(maxlen=ReplayMemory)

	def push(self, state, M,iteration):

		n=M.shape[0]
		for i in range(n):
			#self.buffer.append(( (state[0][i].detach(),state[1][i].detach(),state[2][i].detach(),state[3][i].detach()), M[i].detach(), iteration[i].detach()))
			self.buffer.append(( (state[0][i].detach().cpu(),state[1][i].detach().cpu(),state[2][i].detach().cpu(),state[3][i].detach().cpu()), M[i].detach().cpu(), iteration[i].detach().cpu()))


	def sample(self,batch_size):

		state_t, M_t, iteration_t= zip(*random.sample(self.buffer, batch_size))
		DIM=M_t[0].shape[0]
		outputDIM=M_t[0].shape[1]

		M=(torch.randn(batch_size,DIM, outputDIM)).cuda()
		state = (torch.zeros(batch_size,DIM,outputDIM).cuda(),
								torch.zeros(batch_size,DIM,outputDIM).cuda(),
								torch.zeros(batch_size,DIM,outputDIM).cuda(),
								torch.zeros(batch_size,DIM,outputDIM).cuda(),
								) 
		iteration=torch.zeros(batch_size)

		for i in range(batch_size):
			M[i]=M_t[i].cuda()
			iteration[i]=iteration_t[i].cuda()
			state[0][i]=state_t[i][0].cuda()
			state[1][i]=state_t[i][1].cuda()
			state[2][i]=state_t[i][2].cuda()
			state[3][i]=state_t[i][3].cuda()

		#M.requires_grad=True

		return state, M,iteration

	def shuffle(self):
		
		random.shuffle(self.buffer)

	def __len__(self):
		return len(self.buffer)
