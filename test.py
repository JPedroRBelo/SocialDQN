from utils.misc import *
# Config
#from agent.DoubleQLearner import Agent
from agent.MultimodalNeuralQLearner import MultimodalAgent
from agent.ExperienceReplay import MultimodalReplayBuffer
from config.hyperparams import *
from environment.SimEnvironment import Environment

import matplotlib.pyplot as plt
import threading
from PIL import Image
import sys
import time
from torchvision.utils import save_image


def save_image_thread(ep,step,name,images):
    dirr = 'images/'+str(ep)+"/"

    os.makedirs(dirr, exist_ok=True)
    for i in range(len(images[0])):

        #img = Image.fromarray(images[i])
        #img.save(dirr+str(name)+"_"+str(i)+".png")
        save_image(images[0][i], dirr+name+str(step)+"_"+str(i)+".png")


def plot(save=False):
    df = pandas.DataFrame(scores,columns=['scores','average_scores','std'])
    fig = plt.figure(num=2,figsize=(10, 5))
    plt.clf()
    ax = fig.add_subplot(111)
    episode = np.arange(len(scores))
    plt.plot(episode,df['average_scores'])
    plt.fill_between(episode,df['average_scores'].add(df['std']),df['average_scores'].sub(df['std']),alpha=0.3)
    plt.title(env_name)
    ax.legend([agent.name + ' [ Average scores ]'])
    plt.ylabel('Score')
    plt.xlabel('Episode')
    #plt.show(block=False)
    if(save):
        fig.savefig('scores/%s_%s_batch_%d_lr_%.E_trained_%d_episodes.png'% (agent.name,env_name,params['batch_size'],params['learning_rate'],i_episode))   # save the figure to file

    plt.pause(0.001)  # pause a bit so that plots are updated




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
agent = MultimodalAgent(state_size=state_size, action_size=action_size, param=params, seed=0,)
# Load the pre-trained network

folder = ''
if(len(sys.argv)>1):
    folder = str(sys.argv[1])

agent.import_network(folder+'models/%s_%s'% (agent.name,env_name))

# Define parameters for test
episodes = 1                        # maximum number of test episodes

""" Test loop  """
for i_episode in range(1, episodes+1):
    # Reset the environment
    env.reset()
    



    # Capture the current state
    gray_state,depth_state,num_faces = env.get_screen()


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

        next_gray_state,next_depth_state,num_faces= env.get_screen()
        print(num_faces)
        save_images = False
        if(save_images):
            gray_thread = threading.Thread(target=save_image_thread, args=(i_episode,step,'gray',next_gray_state))
            depth_thread = threading.Thread(target=save_image_thread, args=(i_episode,step,'depth',next_depth_state))
            gray_thread.start()
            depth_thread.start()

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