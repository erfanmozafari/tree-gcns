import numpy as np
import matplotlib.pyplot as plt
from skan import csr
from skimage import io, color, morphology
import networkx as nx
import json
from skimage import io, color, morphology
from skimage.morphology import binary_dilation, binary_erosion, binary_closing

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

    N = 20
    significant_branches = [branches[i] for i in sorted_indices[:N]]
    
    points = []
    G = nx.Graph()
    for i, branch in enumerate(significant_branches):
        head = [float(branch[0, 1]), float(branch[0, 0]), 0.0]
        tail = [float(branch[-1, 1]), float(branch[-1, 0]), 0.0]
        
        G.add_node(int(head[0] * head[1]), pos=head)
        G.add_node(int(tail[0] * tail[1]), pos=tail)
    # if '1' in img_path:
    #     plt.title('Cleaned Skeleton with Top {} Significant Branches'.format(N))
    #     plt.show()

    # # Flatten the skeleton to get all foreground pixel coordinates
    # skeleton_coords = np.column_stack(np.where(skeleton))

    # # Calculate the number of points to sample (99% of all points)
    # num_sample = int(len(skeleton_coords) * 0.3)

    # # Randomly sample the points
    # sampled_coords = skeleton_coords[np.random.choice(len(skeleton_coords), num_sample, replace=False)]

    # G = nx.Graph()

    # # Add points to the graph
    # for coord in sampled_coords:
    #     t = [float(coord[1]), float(coord[0]), float(0)]
    #     G.add_node(int(t[0]*t[1]), pos=(t))

    G.graph['y'] = y
    
    return G


import os

# Define the directory
dir_name = "trees"

# clouds = []
label=0
if os.path.exists(dir_name):
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            print(root, label)
        #     tree = root+'/'+file
        #     G = get_graph(tree, label)
        #     clouds.append(nx.node_link_data(G))
        #     print(root+'/'+file, label)
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

# with open('skeletons-junctions.json', 'w') as f:
#     json.dump(clouds, f)



    