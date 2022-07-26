from utils.misc import *
import math
import random
import time

import matplotlib.pyplot as plt


def plot(save=False):
    df = pandas.DataFrame(scores,columns=['scores','average_scores','std'])
    fig = plt.figure(num=2,figsize=(10, 5))
    plt.clf()


    ax = fig.add_subplot(111)
    episode = np.arange(len(scores))
    plt.plot(episode,df['average_scores'])
    plt.fill_between(episode,df['average_scores'].add(df['std']),df['average_scores'].sub(df['std']),alpha=0.3)
    

    ax.legend([' [ Average scores ]'])
    plt.ylabel('Score')
    plt.xlabel('Episode')
    if(df['average_scores'].size<=1):
        max_total_fails = -6.0
    else:
        max_total_fails = min(df['scores'].min(),df['average_scores'].min())
    if max_total_fails < 0:
         max_total_fails = int(math.floor(max_total_fails))
         max_total_success = 1.5 
    else:
        max_total_fails = int(math.ceil(max_total_fails)) 
        max_total_success = 1.1

    print('\n'+str(episode[-1]))
    print('scores \t\t'+str(df['scores'].min()))
    print('average scores \t'+str(df['average_scores'].min()))
    print('std\t\t'+str(df['std'].min()))
   
    major_ticks = np.arange(max_total_fails, max_total_success, 0.5)
    minor_ticks = np.arange(max_total_fails, max_total_success, 0.1)

    #ax.set_xticks(major_ticks)
    #ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.axhline(1, color='gray', linewidth=0.5)
    ax.axhline(0, color='gray', linewidth=0.5)
    # And a corresponding grid
    #ax.grid(which='both')#,axis='y')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    #plt.show(block=False)
    
    plt.pause(0.001)  # pause a bit so that plots are updated


scores_window = []
scores = []
for i in range(3000):
    time.sleep(1)
    score = random.uniform(-3.0, 1.0)
    scores_window.append(score)
    scores.append([score, np.mean(scores_window), np.std(scores_window)])
    plot()