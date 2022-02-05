from utils.misc import *
import math
# Config
#from agent.DoubleQLearner import Agent
from agent.SocialNQLearner import Agent
from agent.ExperienceReplay import ReplayBuffer
from config.hyperparams import *
from environment.SimEnvironment import Environment

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
    parser.add_argument('-m','--model',default='',type=dir_path)
    return parser.parse_args()

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


def save_action_reward_history(actions_rewards):
    dirr = 'scores/'
    file = 'action_reward_history.dat'
    torch.save(actions_rewards,dirr+file)

def save_social_signals_states(social_signals):
    dirr = 'scores/'
    file = 'social_signals_history.dat'
    torch.save(social_signals,dirr+file)


def save_image_thread(ep,step,name,images):
    dirr = 'images/'+str(ep)+"/"

    os.makedirs(dirr, exist_ok=True)
    for i in range(len(images[0])):

        #img = Image.fromarray(images[i])
        #img.save(dirr+str(name)+"_"+str(i)+".png")
        save_image(images[0][i], dirr+name+str(step)+"_"+str(i)+".png")


def plot(scores,name,params,i_episode,save=False):
    df = pandas.DataFrame(scores,columns=['scores','average_scores','std'])
    fig = plt.figure(num=2,figsize=(10, 5))
    plt.clf()


    ax = fig.add_subplot(111)
    episode = np.arange(len(scores))
    plt.plot(episode,df['average_scores'])
    plt.fill_between(episode,df['average_scores'].add(df['std']),df['average_scores'].sub(df['std']),alpha=0.3)
    plt.title(params['env_name'])
    ax.legend([name + ' [ Average scores ]'])
    plt.ylabel('Score')
    plt.xlabel('Episode')
    if(df['average_scores'].size<=1):
        max_total_fails = params['hs_fail_reward']*params['t_steps']
    else:
        max_total_fails = min(df['scores'].min(),df['average_scores'].min())
    if max_total_fails < 0:
         max_total_fails = int(math.floor(max_total_fails))
         max_total_success = 1.5 
    else:
        max_total_fails = int(math.ceil(max_total_fails)) 
        max_total_success = 1.1

   
    major_ticks = np.arange(max_total_fails, max_total_success, 0.5)
    minor_ticks = np.arange(max_total_fails, max_total_success, 0.1)

    #ax.set_xticks(major_ticks)
    #ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.axhline(1, color='gray', linewidth=0.5)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axhline(-1, color='gray', linewidth=0.5)
    # And a corresponding grid
    #ax.grid(which='both')#,axis='y')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    if(save):
        fig.savefig('scores/%s_%s_batch_%d_lr_%.E_trained_%d_episodes.png'% (name,params['env_name'],params['batch_size'],params['learning_rate'],i_episode))   # save the figure to file

    plt.pause(0.001)  # pause a bit so that plots are updated

def validate_eps(eps=1):  

    # Initialize environment object
    parsed_args = parse_arguments()
    model_dir = parsed_args.model

    spec=importlib.util.spec_from_file_location("cfg",os.path.join(model_dir,"hyperparams.py"))
    cfg =  importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    params = cfg.PARAMETERS['SimDRLSR']
    #check_consistency_in_configuration_parameters(params)
    env_name = params['env_name']

    params['simulation_speed'] = 5



    save_social_states = params['save_social_states']

    save_images = params['save_images']
    solved_score = params['solved_score']
    stop_when_solved = params['stop_when_solved']

    start_simulator = False
    if(not parsed_args.sim==''):
        start_simulator = True
    env = Environment(params,simulator_path=parsed_args.sim,start_simulator=start_simulator)


    # Reset the environment
    #env_info = env.reset()

    # Get environment parameter
    number_of_agents = params['number_of_agents']
    action_size = params['action_size']
    state_size = params['state_size']
    train_after_episodes = params['train_after_episodes']

    print('Number of agents  : ', number_of_agents)
    print('Number of actions : ', action_size)
    print('Dimension of state space : ', state_size)

    # Initialize agent
    agent = Agent(state_size=state_size, action_size=action_size, param=params, seed=0,)

    agent.import_network(os.path.join(model_dir,'models','%s_%s'% (agent.name,env_name)))

    # Initialize replay buffer
    memory = ReplayBuffer(action_size, params['replay_size'], params['batch_size'], seed=0,device=params['device'])
    update_interval = params['update_interval']
    replay_start = params['replay_initial']



    stop_scores = params['stop_scores']
    scores_window_size = params['scores_window_size']

    # Define parameters for e-Greedy policy
    epsilon = params['epsilon_start']           # starting value of epsilon
    epsilon_floor = params['epsilon_final']     # minimum value of epsilon
    epsilon_decay = params['epsilon_decay']     # factor for decreasing epsilon

    """ Training loop  """
    scores = []                                 # list containing scores from each episode
    scores_window = deque(maxlen=scores_window_size)   # last (window_size) scores
    actions_rewards = []
    social_signals = []

    episodes = eps

    for i_episode in range(1, episodes+1):
        ep_actions_rewards = []
        ep_social_state = []
        # Reset the environment
        env.reset()

        # Capture the current state
        gray_state,_ = env.get_screen()

        # Reset score collector
        score = 0
        done = False

        while not done:
            # Action selection by Epsilon-Greedy policy
            action = agent.greedy(gray_state)
            
            reward, done = env.execute(action)
            next_gray_state,_ = env.get_screen()

            if(save_images):
                if(next_gray_state!=None):
                    gray_thread = threading.Thread(target=save_image_thread, args=(i_episode,step,'gray',next_gray_state[0]))
                    gray_thread.start()

            ep_actions_rewards.append([action,reward])
            if(save_social_states):
                ep_social_state.append(gray_state[1])


            # State transition
            gray_state = next_gray_state

            # Update total score
            score += reward
            

        # Push to score list
        actions_rewards.append( ep_actions_rewards)
        if(save_social_states):
            social_signals.append(ep_social_state)
        scores_window.append(score)
        scores.append([score, np.mean(scores_window), np.std(scores_window)])
        plot(scores,agent.name,params,i_episode,save=False)

        # Print episode summary
        print('\r#TRAIN Episode:{}, Score:{:.2f}, Average Score:{:.2f}, Exploration:{:1.4f}'.format(i_episode, score, np.mean(scores_window), epsilon), end="")
        if i_episode % 100 == 0:
            print('\r#TRAIN Episode:{}, Score:{:.2f}, Average Score:{:.2f}, Exploration:{:1.4f}'.format(i_episode, score, np.mean(scores_window), epsilon))
            #agent.export_network('models/%s_%s_ep%s'% (agent.name,env_name,str(i_episode)))
            #agent.export_network('models/%s_%s_%s'% (agent.name,env_name,str(i_episode)))
        if (np.mean(scores_window)>=solved_score)and stop_when_solved:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            #agent.export_network('models/%s_%s_%s'% (agent.name,env_name,str(i_episode)))
            #pass
            break                


    '''
    # Export scores to csv file

    df = pandas.DataFrame(scores,columns=['scores','average_scores','std'])
    df.to_csv('scores/%s_%s_batch_%d_lr_%.E_trained_%d_episodes.csv'% (agent.name,env_name,params['batch_size'],params['learning_rate'],i_episode), sep=',',index=False)
    save_action_reward_history(actions_rewards)
    if(save_social_states):
        save_social_signals_states(social_signals)
    agent.export_network('models/%s_%s'% (agent.name,env_name))
    '''
    plot(scores,agent.name,params,episodes+1,save=True)

    # Close environment    
    env.close_connection()

def just_run(steps=30,restart_done=True):   

    # Initialize environment object
    parsed_args = parse_arguments()
    model_dir = parsed_args.model

    spec=importlib.util.spec_from_file_location("cfg",os.path.join(model_dir,"hyperparams.py"))
    cfg =  importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)


    params = customized_params(cfg.PARAMETERS['SimDRLSR'])
    #check_consistency_in_configuration_parameters(params)
    env_name = params['env_name']

    save_social_states = params['save_social_states']

    save_images = params['save_images']
    solved_score = params['solved_score']
    stop_when_solved = params['stop_when_solved']

    start_simulator = False
    if(not parsed_args.sim==''):
        start_simulator = True
    env = Environment(params,simulator_path=parsed_args.sim,start_simulator=start_simulator)


    # Reset the environment
    #env_info = env.reset()

    # Get environment parameter
    number_of_agents = params['number_of_agents']
    action_size = params['action_size']
    state_size = params['state_size']
    train_after_episodes = params['train_after_episodes']

    print('Number of agents  : ', number_of_agents)
    print('Number of actions : ', action_size)
    print('Dimension of state space : ', state_size)

    # Initialize agent
    agent = Agent(state_size=state_size, action_size=action_size, param=params, seed=0,)

    agent.import_network(os.path.join(model_dir,'models','%s_%s'% (agent.name,env_name)))

    # Initialize replay buffer
    memory = ReplayBuffer(action_size, params['replay_size'], params['batch_size'], seed=0,device=params['device'])
    update_interval = params['update_interval']
    replay_start = params['replay_initial']



    stop_scores = params['stop_scores']
    scores_window_size = params['scores_window_size']

    # Define parameters for e-Greedy policy
    epsilon = params['epsilon_start']           # starting value of epsilon
    epsilon_floor = params['epsilon_final']     # minimum value of epsilon
    epsilon_decay = params['epsilon_decay']     # factor for decreasing epsilon

    """ Training loop  """
    scores = []                                 # list containing scores from each episode
    scores_window = deque(maxlen=scores_window_size)   # last (window_size) scores
    actions_rewards = []
    social_signals = []

    ep_actions_rewards = []
    ep_social_state = []

    done = False
    env.reset()

    # Capture the current state
    gray_state,_ = env.get_screen()


    for step in range(1, steps+1):

        # Reset the environment


        # Reset score collector
        score = 0
        done = False

        # Action selection by Epsilon-Greedy policy
        action = agent.greedy(gray_state)
        
        reward, done = env.execute(action)
        next_gray_state,_ = env.get_screen()

        if(save_images):
            if(gray_state!=None):
                gray_thread = threading.Thread(target=save_image_thread, args=(1,step,'gray',gray_state[0]))
                gray_thread.start()

        ep_actions_rewards.append([action,reward])
        if(save_social_states):
            ep_social_state.append(gray_state[1])


        # State transition
        gray_state = next_gray_state

        # Update total score
        score += reward
        if(done):
            done = False
            env.reset()
            # Capture the current state
            gray_state,_ = env.get_screen()

            

    # Push to score list
    actions_rewards.append( ep_actions_rewards)
    if(save_social_states):
        social_signals.append(ep_social_state)

    save_action_reward_history(actions_rewards)
    if(save_social_states):
        save_social_signals_states(social_signals)
    # Close environment    
    env.close_connection()




def customized_params(params):
    params['screen_width'] = 320
    params['screen_height'] = 240
    params['simulation_speed'] = 1
    params['save_social_states'] = True
    params['save_images'] = True
    params['socket_time_out'] =20.0

    return params 




    

def main():
    #validate_eps(1)
    just_run(steps=100)

if __name__ == "__main__":      
    main()
