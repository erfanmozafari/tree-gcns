import numpy as np
import matplotlib.pyplot as plt
from skan import csr
from skimage import io, color, morphology
import networkx as nx
import json
from skimage import io, color, morphology


def get_graph(img_path, y):
    img = io.imread(img_path)
    if img.shape[2] == 4:
        alpha_threshold = 0.01  # Define a threshold for alpha values
        nearly_transparent = img[:, :, 3] < alpha_threshold * 255
        img[nearly_transparent] = [0, 0, 0, 255]  # Set the color to red with full opacity

    if img.shape[2] == 4:
        img = img[:, :, :3]

    img_gray = color.rgb2gray(img)

    threshold = np.mean(img_gray)
    img_binary = img_gray > threshold

    skeleton = morphology.skeletonize(img_binary)

    # Flatten the skeleton to get all foreground pixel coordinates
    skeleton_coords = np.column_stack(np.where(skeleton))

    # Calculate the number of points to sample (99% of all points)
    num_sample = int(len(skeleton_coords) * 0.3)

    # Randomly sample the points
    sampled_coords = skeleton_coords[np.random.choice(len(skeleton_coords), num_sample, replace=False)]

    G = nx.Graph()

    # Add points to the graph
    for coord in sampled_coords:
        t = [float(coord[1]), float(coord[0]), float(0)]
        G.add_node(int(t[0]*t[1]), pos=(t))

    G.graph['y'] = y
    
    return G


import os

# Define the directory
dir_name = "./trees"

clouds = []
label=0
if os.path.exists(dir_name):
    for root, dirs, files in os.walk(dir_name):
        
        for file in files:
            tree = root+'/'+file
            G = get_graph(tree, label)
            clouds.append(nx.node_link_data(G))
            print(root+'/'+file, label)
        label += 1
else:
    print(f"The directory {dir_name} does not exist.")

# clouds = []
# for i in range(1,50):
#     path = 'maples/maple' + str(i) + '.png'
#     G = do(path)
#     clouds.append(nx.node_link_data(G))
#     print(i)

# Convert numpy int64 to Python int before serialization
# def convert(o):
#     if isinstance(o, np.int64): return int(o)  
#     if isinstance(o, np.float64): return float(o)  
#     raise TypeError

with open('skeletons-30.json', 'w') as f:
    json.dump(clouds, f)



    