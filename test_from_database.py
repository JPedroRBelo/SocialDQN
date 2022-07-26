from utils.misc import *
import math
# Config
#from agent.DoubleQLearner import Agent
from agent.SocialNQLearner import Agent
from agent.ExperienceReplay import ReplayBuffer
from config.hyperparams import *
from environment.DatabaseEnvironment import Environment

import matplotlib.pyplot as plt
import threading
import shutil
import argparse
from PIL import Image
from torchvision.utils import save_image
from datetime import datetime

import importlib.util
from utils.print_colors import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    parser.add_argument('-s','--sim',default='')
    parser.add_argument('-m','--model',default='')
    parser.add_argument('-w','--write',default=False,type=bool)
    parser.add_argument('-a','--alg',default='greedy')
    return parser.parse_args()

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


def save_action_reward_history(path,actions_rewards):
    dirr = path
    file = '/test_action_reward_history.npy'
    np.save(dirr+file,actions_rewards)

def save_social_signals_states(social_signals):
    dirr = 'scores/'
    file = 'test_social_signals_history.dat'
    torch.save(social_signals,dirr+file)


def validate_eps(eps=1):  

    # Initialize environment object
    parsed_args = parse_arguments()
    model_dir = parsed_args.model
    save_results = parsed_args.write

    spec=importlib.util.spec_from_file_location("cfg",os.path.join(model_dir,"hyperparams.py"))
    cfg =  importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    params = cfg.PARAMETERS['SimDRLSR']
    #check_consistency_in_configuration_parameters(params)
    env_name = params['env_name']




    save_social_states = params['save_social_states']

    # Reset the environment
    #env_info = env.reset()

    action_size = params['action_size']
    state_size = params['state_size']


    # Initialize agent
    agent = Agent(state_size=state_size, action_size=action_size, param=params, seed=0,)

    agent.import_network(os.path.join(model_dir,'models','%s_%s'% (agent.name,env_name)))

    """ Training loop  """
    scores = []                                 # list containing scores from each episode
    actions_rewards = []
    social_signals = []

    episodes = eps
    path = '../validation_tool_socialdqn/dataset/'
    env = Environment(params,'../validation_tool_socialdqn/dataset/')
    ep_actions_rewards = []
    for i_episode in range(1, episodes+1):
        
        ep_social_state = []
        # Reset the environment

        # Capture the current state
        gray_state,_ = env.get_screen(i_episode)


        # Action selection by Epsilon-Greedy policy
        action = agent.greedy(gray_state)
        print(action)
        reward = 0
        ep_actions_rewards.append([action,reward])

        
    #save_action_reward_history(os.path.join(path,'scores'),ep_actions_rewards)


def from_images_without_emotions(eps=1):  

    # Initialize environment object
    parsed_args = parse_arguments()
    model_dir = parsed_args.model
    save_results = parsed_args.write
    if(os.path.isfile(model_dir)):
        model_file = model_dir
        model_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(model_dir)), os.pardir))
    else:
        model_file = ''

    spec=importlib.util.spec_from_file_location("cfg",os.path.join(model_dir,"hyperparams.py"))
    cfg =  importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    params = cfg.PARAMETERS['SimDRLSR']
    #check_consistency_in_configuration_parameters(params)
    env_name = params['env_name']




    #save_social_states = params['save_social_states']

    # Reset the environment
    #env_info = env.reset()

    action_size = params['action_size']
    state_size = params['state_size']


    # Initialize agent
    agent = Agent(state_size=state_size, action_size=action_size, param=params, seed=0,)

    if(model_file==''):
        model_file = os.path.join(model_dir,'models','%s_%s'% (agent.name,env_name))
    else:
        model_file = model_file.replace('.pth','')

    agent.import_network(model_file)

    """ Training loop  """
    scores = []                                 # list containing scores from each episode
    actions_rewards = []
    social_signals = []

    episodes = eps
    path = ''
    env = Environment(params,'')
    ep_actions_rewards = []
    for i_episode in range(1, episodes+1):

        print(f'Ep: \t{i_episode}')
        for emotion in range(4):
        
            ep_social_state = []
            # Reset the environment

            # Capture the current state
            gray_state,_ = env.get_screen(i_episode,emotion)


            # Action selection by Epsilon-Greedy policy
            action = agent.greedy(gray_state)
            
            print(f'Emotion:{emotion}. Action: {action}')
            reward = 0
            ep_actions_rewards.append([action,reward])
        print('\n')
        
    #save_action_reward_history(os.path.join(path,'scores'),ep_actions_rewards)


    




def customized_params(params,save_results):
    params['screen_width'] = 1080
    params['screen_height'] = 768
    params['simulation_speed'] = 10
    params['save_social_states'] = save_results
    params['save_action_reward_history'] = save_results
    params['save_images'] = save_results
    params['socket_time_out'] =20.0

    return params 



def main():
    #validate_eps(1)
    from_images_without_emotions(100)

if __name__ == "__main__":      
    main()
