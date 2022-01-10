
# define example
#data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']

'''
data = ['no_face','neutral','positive','negative']


import pandas as pd
s = pd.Series(data)
a = pd.get_dummies(s)
print(a['no_face'].values)

'''
from utils.print_colors import *

error('Testando')
blue('Teste azul')
header('header')


emotional_states =  ['no_face','neutral','positive','negative']
facial_states = ['no_face','face']

n = 4
face = 'positive'
result = 'none'

if(n==2):
	if(face in emotional_states):
		aux = min(emotional_states.index(face),(n-1))
		blue(str(aux))
		result = facial_states[aux]
		
	elif(face in facial_states):
		result = face
	else:
		error('error')

else:
	if(face in emotional_states):
		result = face


print(result)
