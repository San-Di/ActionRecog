import numpy as np

batch_size  = 2
dim = (64,64,3)
n_channels = 3

list_IDs_temp = [1,2,3,4,5,5]
labels = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5 }

X = np.empty((batch_size, *dim))
y = np.empty((batch_size), dtype=int)

# Generate data
for i, ID in enumerate(list_IDs_temp):
    # Store sample
    X[i,] = np.zeros(dim)
    print(np.asarray(X[i,]).shape)

    # Store class
    y[i] = labels[ID]


print(X.shape , y.shape)