import torch
import numpy as np
import pickle
import sys




	


datContent = []


def remove(row):
	folder = 'files'
	rewards=torch.load(folder+'/reward_history.dat')#.detach().cpu().numpy()
	actions=torch.load(folder+'/action_history.dat')#.detach().cpu().numpy()
	recent_rewards=torch.load('recent_rewards.dat')#.detach().cpu().numpy()
	recent_actions=torch.load('recent_actions.dat')#.detach().cpu().numpy()
	ep_rewards=torch.load(folder+'/ep_rewards.dat')



	#torch.save(rewards[row],folder+'/bkup_rewards.dat'+str(row))
	#torch.save(actions[row],folder+'/bkup_actions.dat'+str(row))
	print(len(rewards))
	print(len(actions))
	print(len(ep_rewards))

	rewards = list(rewards)
	actions = list(actions)

	#rewards.append(recent_rewards)
	#actions.append(recent_actions)
	rewards.pop(-1)
	actions.pop(-1)
	ep_rewards.pop(-1)

	print(len(rewards))
	print(len(actions))
	print(len(ep_rewards))
	

	#new_actions = np.delete(actions, row, 0)
	#total_reward = 0
	#for i in recent_rewards:
	#	total_reward += i
	#ep_rewards.append(total_reward)


	#torch.save(ep_rewards,folder+'/ep_rewards.dat')
	#torch.save(rewards,folder+'/reward_history.dat')
	#torch.save(actions,folder+'/action_history.dat')




if len(sys.argv) > 1:
	row = int(sys.argv[1])
	print('Removing row: ',row)
	remove(row)

	


