import torch
import numpy as np


a = np.array([0.63423,-0.456,0.0145,0.5])
b = np.array([-0.3435,-0.456,0.1235,0.151])

#norm_a = a / np.sqrt(np.sum(a**2))
#norm_b = b / np.sqrt(np.sum(b**2))


def normalize(values):
	norm = [(float(i)-min(values))/(max(values)-min(values)) for i in values]
	return (np.array(norm))


norm_a = normalize(a)
norm_b = normalize(b)
print(norm_a)
print(norm_b)
q_fus=((norm_a)*0.5)+((norm_b)*0.5)
print(q_fus)
q_fus = torch.tensor(q_fus.astype(np.float32))


action = np.argmax(q_fus.cpu().data.numpy())
print(action)