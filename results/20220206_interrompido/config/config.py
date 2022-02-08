#datageneration
simulation_speed = 3
t_steps = 50
#environment
raw_frame_height = 320
raw_frame_width = 240
proc_frame_size = 198
state_size = 8
port = 12375        
#host='192.168.0.11'
#host='10.62.6.208'
host='127.0.0.1'
#NQL
actions_names = ['Wait','Look','Wave','Handshake']
actions	= ['1','2','3','4']
n_actions = len(actions)
#epsilon annealing
gamma = 0.999
eps_start   = 0.99
eps_end	 = 0.1
ep_endt_number = 50
#ep_endt	= ep_endt_number * t_steps
learn_start= 0

#trainNQL
device = "cpu"#cuda
t_eps = 10000
batch_size = 128
discount       = 0.99 #Discount factor.
replay_memory_size  = 3000
eps_decay = 10000
target_q       = 4

#rewards
neutral_reward = 0
##handshake
hs_success_reward = 1
hs_fail_reward = -0.1
##eyegaze
eg_success_reward = 0
eg_fail_reward = -0.1
##smile

#fail episode
ep_fail_reward = -1 


#network
noutputs=4
nfeats=8
nstates=[16,32,64,256]
#kernels = [4,2]
kernels = [9,5]
strides = [3,1]
poolsize=2

