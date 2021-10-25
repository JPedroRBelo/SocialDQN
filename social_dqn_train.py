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


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    parser.add_argument('-sim',default='')

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


def erase_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def delete_old_files():
   erase_folder('images')
   erase_folder('scores')
   erase_folder('models')


def init_files():
    
    Path("images").mkdir(parents=True, exist_ok=True)
    Path("scores").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)

def save_train_files(cfg):
    now = datetime.now()
    filename = now.strftime("%Y%m%d_%H%M%S")
    folder = os.path.join('results', filename)
    Path(folder).mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg.__file__,folder)
    shutil.copytree('scores',os.path.join(folder, 'scores'), dirs_exist_ok=True)
    shutil.copytree('models',os.path.join(folder, 'models'), dirs_exist_ok=True)




def main(cfg):
    # Initialize environment object
    parsed_args = parse_arguments()
    params = cfg.PARAMETERS['SimDRLSR']
    env_name = params['env_name']



    save_images = params['save_images']
    solved_score = params['solved_score']

    start_simulator = False
    if(not parsed_args.sim==''):
        start_simulator = True
    env = Environment(params,simulator_path=parsed_args.sim,start_simulator=start_simulator)

    # Reset the environment
    env_info = env.reset()

    # Get environment parameter
    number_of_agents = params['number_of_agents']
    action_size = params['action_size']
    state_size = params['state_size']

    print('Number of agents  : ', number_of_agents)
    print('Number of actions : ', action_size)
    print('Dimension of state space : ', state_size)

    # Initialize agent
    agent = Agent(state_size=state_size, action_size=action_size, param=params, seed=0,)

    # Initialize replay buffer
    memory = ReplayBuffer(action_size, params['replay_size'], params['batch_size'], seed=0,device=params['device'])
    update_interval = params['update_interval']
    replay_start = params['replay_initial']

    # Define parameters for training
    episodes = params['train_episodes']         # maximum number of training episodes
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

    for i_episode in range(1, episodes+1):
        ep_actions_rewards = []
        # Reset the environment
        env.reset()

        # Capture the current state
        gray_state,depth_state = env.get_screen()

        # Reset score collector
        score = 0
        done = False
        # One episode loop
        step = 0
        while not done:
            # Action selection by Epsilon-Greedy policy
            action = agent.eGreedy(gray_state,epsilon)
            #action = agent.select_action(gray_state,depth_state)
            
            reward, done = env.execute(action)
            next_gray_state,next_depth_state = env.get_screen()

            if(save_images):
                gray_thread = threading.Thread(target=save_image_thread, args=(i_episode,step,'gray',next_gray_state))
                depth_thread = threading.Thread(target=save_image_thread, args=(i_episode,step,'depth',next_depth_state))
                gray_thread.start()
                depth_thread.start()

            # Store experience
            memory.push(gray_state, action, reward, next_gray_state, done)

            ep_actions_rewards.append([action,reward])
            # Update Q-Learning
            step += 1
            if (step % update_interval) == 0 and len(memory) > replay_start:
                # Recall experiences (miniBatch)
                experiences = memory.recall()
                # Train agent
                agent.learn(experiences)
                #print('\r#Training step:{}'.format(step), end="")

            # State transition
            gray_state = next_gray_state
            depth_state = next_depth_state

            # Update total score
            score += reward

        # Push to score list
        actions_rewards.append( ep_actions_rewards)
        scores_window.append(score)
        scores.append([score, np.mean(scores_window), np.std(scores_window)])
        plot(scores,agent.name,params,i_episode,save=False)

        # Print episode summary
        print('\r#TRAIN Episode:{}, Score:{:.2f}, Average Score:{:.2f}, Exploration:{:1.4f}'.format(i_episode, score, np.mean(scores_window), epsilon), end="")
        if i_episode % 100 == 0:
            print('\r#TRAIN Episode:{}, Score:{:.2f}, Average Score:{:.2f}, Exploration:{:1.4f}'.format(i_episode, score, np.mean(scores_window), epsilon))
            #agent.export_network('models/%s_%s_ep%s'% (agent.name,env_name,str(i_episode)))
            #agent.export_network('models/%s_%s_%s'% (agent.name,env_name,str(i_episode)))
        if np.mean(scores_window)>=solved_score:
            #print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            #agent.export_network('models/%s_%s_%s'% (agent.name,env_name,str(i_episode)))
            pass
            #break
                


        # Update exploration
        epsilon = max(epsilon_floor, epsilon*epsilon_decay)
    """ End of the Training """

    # Export scores to csv file
    df = pandas.DataFrame(scores,columns=['scores','average_scores','std'])
    df.to_csv('scores/%s_%s_batch_%d_lr_%.E_trained_%d_episodes.csv'% (agent.name,env_name,params['batch_size'],params['learning_rate'],i_episode), sep=',',index=False)
    save_action_reward_history(actions_rewards)
    agent.export_network('models/%s_%s'% (agent.name,env_name))
    plot(scores,agent.name,params,episodes+1,save=True)
    # Close environment    
    env.close_connection()



if __name__ == "__main__":    
    init_files()
    delete_old_files()
    import config.test1 as cfg     
    main(cfg)
    save_train_files(cfg)