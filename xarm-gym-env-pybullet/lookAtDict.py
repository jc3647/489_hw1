import numpy as np

# opens and print a .pkl file
def load_pickle(file_name):
    read_dictionary = np.load(file_name,allow_pickle='TRUE').item()
    print(read_dictionary)


load_pickle('greedyHumanPolicy5.npy')