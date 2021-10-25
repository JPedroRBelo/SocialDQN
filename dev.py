from utils.misc import *
# Config
#from agent.DoubleQLearner import Agent
from agent.MultimodalNeuralQLearner import MultimodalAgent
from agent.SocialNeuralQLearner import SocialAgent
from agent.ExperienceReplay import MultimodalReplayBuffer
from config.hyperparams import *
from environment.SimEnvironment import Environment

import matplotlib.pyplot as plt
import threading
from PIL import Image
import sys
import time
from torchvision.utils import save_image





params = PARAMETERS['SimDRLSR']
params['save_images'] = True
env_name = params['env_name']


params['save_images'] = True

save_images = params['save_images']
solved_score = params['solved_score']

params['simulation_speed'] = 1

env = Environment(params,start_simulator=True)

# Get environment parameter
number_of_agents = params['number_of_agents']
action_size = params['action_size']
state_size = params['state_size']

print('Number of agents  : ', number_of_agents)
print('Number of actions : ', action_size)
print('Dimension of state space : ', state_size)

# Initialize agent
agent = SocialAgent(state_size=state_size, action_size=action_size, param=params, seed=0,)
# Load the pre-trained network

folder = ''
if(len(sys.argv)>1):
    folder = str(sys.argv[1])

#agent.import_network(folder+'models/%s_%s'% (agent.name,env_name))

# Define parameters for test
episodes = 1                        # maximum number of test episodes

""" Test loop  """
for i_episode in range(1, episodes+1):
    # Reset the environment
    env.reset() 



    # Capture the current state
    gray_state,depth_state = env.get_screen()


    # Reset score collector
    score = 0
    done = False
    step = 0
    # One episode loop
    while not done:
        # Action selection by Epsilon-Greedy policy
        start = time.time()
        action = agent.greedy(gray_state,depth_state)

        reward, done = env.execute(action)
        print('Ep {}: \nAction:\t {}\nReward:\t {}'.format(step,str(action),str(reward)))

        next_gray_state,next_depth_state= env.get_screen()
    

        # State transition
        gray_state = next_gray_state
        depth_state = next_depth_state

        # Update total score
        score += reward
        step += 1
        end = time.time()
        print("Step Completion time: "+str(end - start))


    # Print episode summary
    print('\r#TEST Episode:{}, Score:{:.2f}'.format(i_episode, score))

""" End of the Test """

# Close environment
env.close_connection()