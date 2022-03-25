from utils.misc import *
import math
from threading import Thread
# Config
#from agent.DoubleQLearner import Agent
from agent.SocialNQLearner import Agent
from agent.MultimodalNeuralQLearner import MultimodalAgent
from agent.ExperienceReplay import ReplayBuffer
from agent.ExperienceReplay import MultimodalReplayBuffer
from config.hyperparams import *
from environment.SimEnvironment import Environment

import matplotlib.pyplot as plt
import threading
import shutil
import argparse
from PIL import Image
from torchvision.utils import save_image
from datetime import datetime

from utils.print_colors import *

import curses
import time

from inspect import currentframe, getframeinfo


debug = False
ep_debug = 0

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    parser.add_argument('-sim',default='')

    return parser.parse_args()

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


def save_action_reward_history(actions_rewards,ep=0):
    filename_ep = str(ep)
    if(ep==0):
        filename_ep=''
    dirr = 'scores/'
    file = 'action_reward_history'+filename_ep+'.dat'
    torch.save(actions_rewards,dirr+file)

def save_social_signals_states(social_signals,ep=0):
    filename_ep = str(ep)
    if(ep==0):
        filename_ep=''
    dirr = 'scores/'
    file = 'social_signals_history'+filename_ep+'.dat'
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
    episode = np.arange(i_episode)
    plt.plot(episode,df['average_scores'][:i_episode])
    plt.fill_between(episode,df['average_scores'][:i_episode].add(df['std'][:i_episode]),df['average_scores'][:i_episode].sub(df['std'][:i_episode]),alpha=0.3)
    plt.title(params['env_name'])
    ax.legend([name + ' [ Average scores ]'])
    plt.ylabel('Score')
    plt.xlabel('Episode')
    if(df['average_scores'][:i_episode].size<=1):
        max_total_fails = params['hs_fail_reward']*params['t_steps']
    else:
        max_total_fails = min(df['scores'][:i_episode].min(),df['average_scores'][:i_episode].min())
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
            error('Failed to delete %s. Reason: %s' % (file_path, e))

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

def save_train_files(cfg,notes=""):
    now = datetime.now()
    filename = now.strftime("%Y%m%d_%H%M%S")
    folder = os.path.join('results', filename)
    Path(folder).mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg.__file__,folder)
    shutil.copytree('scores',os.path.join(folder, 'scores'), dirs_exist_ok=True)
    shutil.copytree('models',os.path.join(folder, 'models'), dirs_exist_ok=True)
    if(not notes==""):
        with open(os.path.join(folder,'notes.txt'), 'w') as f:
            f.write(notes)


def check_consistency_in_configuration_parameters(prm):
    if(prm['save_images']):
         warning("Saving images can cause training latency and consume large disk space.")



def main(cfg):
    envs = []
    try:
        #
        
        # Initialize environment object
        parsed_args = parse_arguments()
        params = cfg.PARAMETERS['SimDRLSR']
        check_consistency_in_configuration_parameters(params)
        env_name = params['env_name']

        solved_score = params['solved_score']
        stop_when_solved = params['stop_when_solved']

        start_simulator = False
        if(not parsed_args.sim==''):
            start_simulator = True

        # Get environment parameter
        number_of_agents = params['number_of_agents']
        action_size = params['action_size']
        state_size = params['state_size']
        train_after_episodes = params['train_after_episodes']
        use_depth_state = params['use_depth_state']
        use_only_depth_state = params['use_only_depth_state']


        print('Number of agents  : ', number_of_agents)
        print('Number of actions : ', action_size)
        print('Dimension of state space : ', state_size)


        if(use_depth_state) and (not use_only_depth_state):
            #Multimodal states
            # Initialize agent
            agent = MultimodalAgent(state_size=state_size, action_size=action_size, param=params, seed=0,)
            # Initialize replay buffer
            memory = MultimodalReplayBuffer(action_size, params['replay_size'], params['batch_size'], seed=0,device=params['device'])
        else:
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
                           
        scores_window = deque(maxlen=scores_window_size)   # last (window_size) scores
        scores = [[0, 0, 0]] * episodes      
        actions_rewards = [None] * episodes  
        social_signals = [None] * episodes  

        for i in range(number_of_agents):
            envs.append(Environment(params,simulator_path=parsed_args.sim,start_simulator=start_simulator,port=params['port']+i))
            #envs[-1].reset()

        threads_agents = [None] * number_of_agents
        threads_times = [0] * number_of_agents
        threads_at_ep = [None] * number_of_agents
        queue_episodes = deque(range(episodes))
        ep_count = 0;
        max_thread_time = ((params['t_steps'] * 15) / params['simulation_speed'])+30

        #stdscr = curses.initscr()

        
        envs_fails = [0] * number_of_agents

        time_init = time.time()
        while ep_count<episodes:

            thread_log = ""
            
            for i in range(len(threads_agents)):
                #envs[i].setDebug(threads_at_ep[i])
                if(threads_agents[i]!=None):
                    alive = "Running" if threads_agents[i].is_alive() else "Dead"
                    #thread_log += ' #THREAD {}: {}'.format(i, alive)
                    thread_alive_time = (time.time() - threads_times[i])
                    thread_log += ' #\nTHREAD {}: {} Time: {}'.format(i, alive,thread_alive_time)
                    if(not threads_agents[i].is_alive()):
                        if(actions_rewards[threads_at_ep[i]]!=None):
                            ep_count+=1
                            epsilon = max(epsilon_floor, epsilon*epsilon_decay)
                            plot(scores,agent.name,params,ep_count,save=False)


                            if (train_after_episodes) and (ep_count % update_interval) == 0 and len(memory) > replay_start:
                                # Recall experiences (miniBatch)
                                experiences = memory.recall()
                                # Train agent
                                agent.learn(experiences)
                                print('\r#Training step:{}'.format(ep_count), end="")

                        else: 
                            queue_episodes.appendleft(threads_at_ep[i])
                            envs_fails[i] += 1
                            if(envs_fails[i]>=5):
                                envs[i].close_connection()
                                envs[i] = Environment(params,simulator_path=parsed_args.sim,start_simulator=start_simulator,port=params['port']+i)
                                time.sleep(2)
                        if(len(queue_episodes)>0):
                            ep_at = queue_episodes.popleft()
                            threads_agents[i] = Thread(target=execute_ep, args=(envs[i],agent,ep_at,memory,params,epsilon,scores,scores_window,actions_rewards,social_signals,i))
                            #threads.append(t)
                            threads_agents[i].setDaemon(True)  
                            threads_agents[i].start()
                            threads_times[i] = time.time()
                            threads_at_ep[i] = ep_at
                        else:
                            threads_agents[i] = None
                    

                    elif(thread_alive_time > max_thread_time):
                        print("#THREAD "+str(i)+" taking too long... ep"+str(threads_at_ep[i])+"... "+str(thread_alive_time)+" seconds alive.")
                        
                        #threads_agents[i].daemon()
                        #envs[i].setDebug(threads_at_ep[i])
                        try:                            
                            result = envs[i].reset()
                        except Exception:
                            result = 0
                            envs[i].close_connection()
                        if(result==0):
                            time.sleep(1)
                            envs[i] = Environment(params,simulator_path=parsed_args.sim,start_simulator=start_simulator,port=params['port']+i)

                        #time.sleep(1)
                        #envs[i] = Environment(params,simulator_path=parsed_args.sim,start_simulator=start_simulator,port=params['port']+i)

                        threads_agents[i] = Thread(target=execute_ep, args=(envs[i],agent,threads_at_ep[i],memory,params,epsilon,scores,scores_window,actions_rewards,social_signals,i))
                        threads_agents[i].setDaemon(True)  
                        threads_agents[i].start()
                        threads_times[i] = time.time()
                        threads_at_ep[i] = ep_at
                        



                else:
                    thread_log += '\n #THREAD{}: NONE'.format(i)
                    if(len(queue_episodes)>0):
                        ep_at = queue_episodes.popleft()
                        threads_agents[i] = Thread(target=execute_ep, args=(envs[i],agent,ep_at,memory,params,epsilon,scores,scores_window,actions_rewards,social_signals))
                        #threads.append(t)
                        threads_agents[i].setDaemon(True)  
                        threads_agents[i].start()
                        threads_times[i] = time.time()
                        threads_at_ep[i] = ep_at
                time_now = time.time() - time_init
                #if(int(time_now)%20==0):
                #   print(thread_log)

            if(ep_count % params['save_interval'] == 0 ) and (ep_count>0):
                # Export scores to csv file
                df = pandas.DataFrame(scores,columns=['scores','average_scores','std'])
                df.to_csv('scores/%s_%s_batch_%d_lr_%.E_trained_%d_episodes.csv'% (agent.name,env_name,params['batch_size'],params['learning_rate'],ep_count), sep=',',index=False)
                save_action_reward_history(actions_rewards,ep_count)
                save_social_states = params['save_social_states']
                if(save_social_states):
                    save_social_signals_states(social_signals,ep_count)
                agent.export_network('models/%s_%s_%s'% (agent.name,env_name,str(ep_count)))
                if(ep_count):
                    plot(scores,agent.name,params,ep_count,save=True)


        for env in envs:  
            env.close_connection()



        # Export scores to csv file
        df = pandas.DataFrame(scores,columns=['scores','average_scores','std'])
        df.to_csv('scores/%s_%s_batch_%d_lr_%.E_trained_%d_episodes.csv'% (agent.name,env_name,params['batch_size'],params['learning_rate'],ep_count), sep=',',index=False)
        save_action_reward_history(actions_rewards)
        save_social_states = params['save_social_states']
        if(save_social_states):
            save_social_signals_states(social_signals)
        agent.export_network('models/%s_%s'% (agent.name,env_name))
        if(ep_count):
            plot(scores,agent.name,params,ep_count,save=True)
        # Close environment  
    except KeyboardInterrupt:
        for env in envs:  
            env.close_connection()
        print("Exiting...")
        sys.exit()



def execute_ep(env,agent,i_episode,memory,params,epsilon,scores,scores_window,actions_rewards,social_signals,id=0):
        train_after_episodes = params['train_after_episodes']
        save_social_states = params['save_social_states']
        save_images = params['save_images']    
        update_interval = params['update_interval']
        ep_actions_rewards = []
        ep_social_state = []
        # Reset the environment
        #env.reset(restart_simulator=False)
        try:                            
            result = env.reset()
        except Exception:
            result = 0
            env.close_connection()
        if(result==0):
            time.sleep(1)
            envs[i] = Environment(params,simulator_path=parsed_args.sim,start_simulator=start_simulator,port=params['port']+i)



        env.episode = i_episode

        # Capture the current state
        try:
            gray_state,depth_state = env.get_screen()
        except:
            print("Cant get states... Thread "+str(id)+" Exiting")
            return 0
            
        # Reset score collector
        score = 0
        done = False
        # One episode loop
        step = 0
        while not done:
            
            # Action selection by Epsilon-Greedy policy
            if(isinstance(agent,Agent)):
                action = agent.eGreedy(gray_state,epsilon)
            elif(isinstance(agent,MultimodalAgent)):
                action = agent.eGreedy(gray_state,depth_state,epsilon)
            else:
                error('Unknown Agent Type!')

            #action = agent.select_action(gray_state,depth_state)    
            try:         
                reward, done = env.execute(action)   
            except:
                print("Cant get reward... Thread "+str(id)+" Exiting")
                return 0        

            try:         
                next_gray_state,next_depth_state = env.get_screen()
            except:
                print("Cant get states... Thread "+str(id)+" Exiting")
                return 0
            

            if(save_images):
                if(next_gray_state!=None):
                    gray_thread = threading.Thread(target=save_image_thread, args=(i_episode,step,'gray',next_gray_state[0]))
                    gray_thread.start()
                if(next_depth_state!=None):
                    depth_thread = threading.Thread(target=save_image_thread, args=(i_episode,step,'depth',next_depth_state[0]))                    
                    depth_thread.start()

            # Store experience
            if(isinstance(memory,ReplayBuffer)):
                memory.push(gray_state, action, reward, next_gray_state, done)
            elif(isinstance(memory,MultimodalReplayBuffer)):
                memory.push(gray_state,depth_state, action, reward, next_gray_state,next_depth_state, done)
            else:
                error('Unknown ReplayBuffer Type!')

            
            ep_actions_rewards.append([action,reward])
            
            if(save_social_states):
                ep_social_state.append(gray_state[1])
            # Update Q-Learning
            step += 1
            
            if (not train_after_episodes) and (step % update_interval) == 0 and len(memory) > replay_start:
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
        

        actions_rewards[i_episode] =  ep_actions_rewards
        if(save_social_states):
            social_signals[i_episode] =  ep_social_state
        scores_window.append(score)
        scores[i_episode] = [score, np.mean(scores_window), np.std(scores_window)]
        
        # Print episode summary
        print('\r#TRAIN Episode:{}, Score:{:.2f}, Average Score:{:.2f}, Exploration:{:1.4f}'.format(i_episode, score, np.mean(scores_window), epsilon), end="")
        #if i_episode % 100 == 0:
            #print('\r#TRAIN Episode:{}, Score:{:.2f}, Average Score:{:.2f}, Exploration:{:1.4f}'.format(i_episode, score, np.mean(scores_window), epsilon))
            #agent.export_network('models/%s_%s_ep%s'% (agent.name,env_name,str(i_episode)))
            #agent.export_network('models/%s_%s_%s'% (agent.name,env_name,str(i_episode)))
        '''
        if (np.mean(scores_window)>=solved_score)and stop_when_solved:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            #agent.export_network('models/%s_%s_%s'% (agent.name,env_name,str(i_episode)))
            #pass
            trained = True;           
        '''     

        


if __name__ == "__main__":    
    init_files()
    delete_old_files()
    import config.hyperparams as cfg     
    main(cfg)
    notes = '###Testing SimDRLSR v0.331####\nSocialDQN\nPure MDQN'
    save_train_files(cfg,notes)