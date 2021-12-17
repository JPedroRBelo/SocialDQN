
# define example
#data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
data = ['no_face','neutral','positive','negative']


import pandas as pd
s = pd.Series(data)
a = pd.get_dummies(s)
print(a['no_face'].values)