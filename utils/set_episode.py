#!/usr/bin/python
import sys
import torch

if len(sys.argv) > 1:
	episode = sys.argv[1]
	print('Configuring files with episode: ',episode)
	torch.save(episode,'files/episode.dat')
	with open('files/episode.txt', 'w') as f:
		f.write(str(episode))

else: 
	print('None')

