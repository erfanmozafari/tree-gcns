from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from skan import csr
from skimage import io, color, morphology
import networkx as nx
import json
from skimage import io, color, morphology
from skimage.morphology import binary_dilation, binary_erosion, binary_closing
import matplotlib.pyplot as plt
import networkx as nx

import networkx as nx
import numpy as np


# Function to find the nearest neighbor of a node
def find_nearest_neighbor(G, node):
    min_distance = np.inf
    nearest_neighbor = None
    x1, y1 = G.nodes[node]['pos']
    for n in G.nodes:
        if n != node and not G.has_edge(node, n):
            x2, y2 = G.nodes[n]['pos']
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if distance < min_distance:
                min_distance = distance
                nearest_neighbor = n
    return nearest_neighbor


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

    selem = np.ones((3, 3))

    skeleton_dilated = binary_dilation(skeleton, selem)
    skeleton_cleaned = binary_erosion(skeleton_dilated, selem)
    skeleton_closed = binary_closing(skeleton_cleaned, selem)

    skeleton_graph = csr.Skeleton(skeleton_closed)
    branches = [skeleton_graph.path_coordinates(i) for i in range(skeleton_graph.paths.indptr.shape[0] - 1)]
    branch_lengths = [np.linalg.norm(branch[-1] - branch[0]) for branch in branches]
    sorted_indices = np.argsort(branch_lengths)[::-1]

    N = 25
    significant_branches = [branches[i] for i in sorted_indices[:N]]
    
    G = nx.Graph()
    for i, branch in enumerate(significant_branches):
        head = [float(branch[0, 1]), float(branch[0, 0])]
        tail = [float(branch[-1, 1]), float(branch[-1, 0])]
        id_head = int(head[0] * head[1])
        id_tail = int(tail[0] * tail[1])
        G.add_node(id_head, pos=head)
        G.add_node(id_tail, pos=tail)

    # Get all positions
    # positions = np.array([data['pos'] for node, data in G.nodes(data=True)])

    # # Calculate the min and max of the positions
    # min_pos = positions.min(axis=0)
    # max_pos = positions.max(axis=0)

    # # Normalize the positions
    # for node, data in G.nodes(data=True):
    #     normalized_pos = (np.array(data['pos']) - min_pos) / (max_pos - min_pos)
    #     G.nodes[node]['pos'] = tuple(normalized_pos)
    
    # Connect each node to its nearest neighbor and set the weight of the edge
    while not nx.is_connected(G):
        for node in G.nodes:
            nearest_neighbor = find_nearest_neighbor(G, node)
            if nearest_neighbor is not None:
                pos1 = np.array(G.nodes[node]['pos'])
                pos2 = np.array(G.nodes[nearest_neighbor]['pos'])
                distance = np.linalg.norm(pos1 - pos2)
                G.add_edge(node, nearest_neighbor, weight=distance)

    # Compute the minimum spanning tree
    T = nx.minimum_spanning_tree(G, weight='weight')

    # Get the positions of the nodes
    pos = nx.get_node_attributes(G, 'pos')

    # for n in T.nodes():
    #     print(T.nodes[n])

    G.graph['y'] = y
    pos = nx.get_node_attributes(T, 'pos')
    # nx.draw(T, pos)
    # plt.show()
    
    return T


get_graph('trees/Maples/1.png', 1)

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
def convert(o):
    if isinstance(o, np.int64): return int(o)  
    if isinstance(o, np.float64): return float(o)  
    raise TypeError

with open('skeletons-mst.json', 'w') as f:
    json.dump(clouds, f)