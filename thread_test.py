from time import sleep, perf_counter
from threading import Thread
from collections import deque
import random



def task(id,ep,results):
    print(f'Thread {id} Starting the task {ep}...')
    time = random.randint(0,2)
    sleep(time)
    results.append(time)
    print(results)
    print(f'Done Thread {id} ep {ep}...')


episodes = deque(range(0,20))
print(episodes)

scores_window = deque(maxlen=2)

scores = [[0,0,0]] * 5
print(scores)

threads = [None] * 5
results = []
start_time = perf_counter()
count = 0
while count < 20:

    #ep = episodes.popleft()

 

    # create and start 10 threads

    for i in range(len(threads)):
        if(threads[i]!=None):
            if(not threads[i].is_alive()):
                count+=1
                if(len(episodes)>0):
                    threads[i] = Thread(target=task, args=(i,episodes.popleft(),scores_window,))
                    #threads.append(t)
                    threads[i].start()
                else:
                    threads[i] = None
        else:
            if(len(episodes)>0):
                threads[i] = Thread(target=task, args=(i,episodes.popleft(),scores_window,))
                #threads.append(t)
                threads[i].start()


    # wait for the threads to complete

end_time = perf_counter()
print(len(results))
print(count)
print(f'It took {end_time- start_time: 0.2f} second(s) to complete.')