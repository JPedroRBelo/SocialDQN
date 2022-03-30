import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import  DataLoader
from torchvision import models

import torchvision.transforms as transforms
import torchvision.datasets as dataset

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from network.SDQNetwork import *
import torch.nn as nn

from utils.misc import *
import math
# Config
#from agent.DoubleQLearner import Agent
from agent.SocialNQLearner import Agent
from agent.ExperienceReplay import ReplayBuffer
from config.hyperparams import *
#from environment.SimEnvironment import Environment
from environment.DatabaseEnvironment import Environment

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
    parser.add_argument('-m','--model',default='',type=dir_path)
    parser.add_argument('-w','--write',default=False,type=bool)
    parser.add_argument('-a','--alg',default='greedy')
    return parser.parse_args()

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")



def view(eps=1):  

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
    model = agent.Q_network

    """ Training loop  """
    scores = []                                 # list containing scores from each episode
    actions_rewards = []
    social_signals = []

    episodes = eps
    path = '../validation_tool_socialdqn/dataset/'
    env = Environment(params,'../validation_tool_socialdqn/dataset/')


    # we will save the conv layer weights in this list
    model_weights =[]
    #we will save the 49 conv layers in this list
    conv_layers = []# get all the model children as list
    model_children = list(model.children())#counter to keep count of the conv layers
    counter = 0#append all the conv layers and their respective wights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter+=1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])

        elif type(model_children[i]) == nn.Sequential:

            for child in model_children[i].children():
                if type(child) == nn.Conv2d:
                    counter+=1
                    model_weights.append(child.weight)
                    conv_layers.append(child)

    #print(model)





    
    ep_actions_rewards = []
    for i_episode in range(1, episodes+1):
        
        ep_social_state = []
        # Reset the environment

        # Capture the current state
        gray_state,_ = env.get_screen(i_episode)
        image = gray_state[0]
        #print(f"Image shape: {image.shape}")


        # Action selection by Epsilon-Greedy policy
        action = agent.greedy(gray_state)
        print(action)
        reward = 0
        outputs = []
        names = []
        for layer in conv_layers[0:]:
            image = layer(image)
            outputs.append(image)
            names.append(str(layer))

        #print(action,reward)
        processed = []
        for feature_map in outputs:
            feature_map = feature_map.squeeze(0)
            gray_scale = torch.sum(feature_map,0)
            gray_scale = gray_scale / feature_map.shape[0]
            processed.append(gray_scale.data.cpu().numpy())

        fig = plt.figure(figsize=(30, 50))
        for i in range(len(processed)):
            a = fig.add_subplot(5, 4, i+1)
            imgplot = plt.imshow(processed[i])
            a.axis("off")
            a.set_title(names[i].split('(')[0], fontsize=30)
        plt.savefig(str('convs/feature_maps'+str(i_episode)+'.jpg'), bbox_inches='tight')




def main():
    view(eps=100)

if __name__ == "__main__":      
    main()
