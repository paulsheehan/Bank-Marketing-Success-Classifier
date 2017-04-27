import pandas as pd
import numpy as np
import math

training_data = pd.read_csv('data/trainingset.txt')

# training_data.iloc[ROW][COL]
# print(training_data.iloc[2][1])

# Prepare data
# Chosen input features
# Age and Balance features
feature_input = training_data.iloc[:, [1, 6]].as_matrix()
result_output = training_data.iloc[:, [17]].as_matrix()

# # For example Case:
f1 = feature_input[0:1000, [0]]
f2 = feature_input[0:1000, [1]]


# Hyperparameters for logistic regression
learning_rate = 0.00001
training_epochs = 10000
display_step = 1000
n_samples = len(result_output)

# initialized as 0 to begin
weigths = np.zeros((n_samples))
biases = np.zeros((n_samples))

def euclidean(v1,v2):
  d=0.0
  for i in range(len(v1)):
    d+=(v1[i]-v2[i])**2
  return math.sqrt(d)

def getdistances(data,vec1):
  distancelist=[]

  # Loop over every item in the dataset
  for i in range(len(data)):
    vec2=data[i]

    # Add the distance and the index
    distancelist.append((euclidean(vec1[i],vec2),i))

  # Sort by distance
  distancelist.sort()
  return distancelist

def knnestimate(data,vec1,k=5):
  # Get sorted distances
  dlist=getdistances(data,vec1)
  avg=0.0

  # Take the average of the top k results
  for i in range(k):
    idx=dlist[i][1]
    avg+=data[idx][[0]]
  avg=avg/k
  return avg

distance = knnestimate(f1, f2)
print(distance)
# print(getdistances(f1, f2))

