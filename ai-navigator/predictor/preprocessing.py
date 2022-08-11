import json
import numpy as np

with open('./dataset/train/run1.json', 'r') as f:
  data = json.load(f)

w_s = 2
t_s = 2

for i in range(len(data)):
    if (len(data) - i > w_s+t_s+1):
        current_position = data[i]
        source = [np.subtract(data[i+acc+1], current_position) for acc in range(w_s)]
        source.append(np.subtract(data[i+w_s+t_s+1], current_position))
        target = [np.subtract(data[i+acc+w_s+1], current_position) for acc in range(t_s)]
        