import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, morphology
from skimage.morphology import binary_dilation, binary_erosion, binary_closing
from skan import csr
import matplotlib.cm as cm
import networkx as nx
import json

def process_image(img_path):
    img = io.imread(img_path)

    if img.shape[2] == 4:
        alpha_threshold = 0.01  # Define a threshold for alpha values
        nearly_transparent = img[:, :, 3] < alpha_threshold * 255
        img[nearly_transparent] = [0, 0, 0, 255]  # Set the color to red with full opacity

    if img.shape[2] == 4:
        # Discard the alpha channel
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

    # print(img_path)
    # if '1' in img_path:
    #     # plt.imshow(img)
    #     plt.figure(figsize=(8, 8))
    #     plt.imshow(skeleton_cleaned, cmap='gray')
    #     # plt.imshow()
    #     plt.show()

    edges = []
    
    plt.figure(figsize=(8, 8))
    # plt.imshow(img)
    for i, branch in enumerate(significant_branches):
        plt.scatter(branch[0, 1], branch[0, 0], c='r')  # Head pixel
        plt.scatter(branch[-1, 1], branch[-1, 0], c='r')  # Tail pixel
        head = (branch[0, 1], branch[0, 0])
        tail = (branch[-1, 1], branch[-1, 0])
        edges.append((head, tail))
    # if '1' in img_path:
    #     plt.title('Cleaned Skeleton with Top {} Significant Branches'.format(N))
    #     plt.show()

    G = nx.Graph()
    edges = [(tuple(edge[0]), tuple(edge[-1])) for edge in edges]

    pos = {node: (np.cos(np.radians(270))*node[0] - np.sin(np.radians(270))*node[1], 
                  np.sin(np.radians(270))*node[0] + np.cos(np.radians(270))*node[1]) for edge in edges for node in edge}

    x_values = [x for x, y in pos.values()]
    y_values = [y for x, y in pos.values()]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)

    pos = {node: ((x - x_min) / (x_max - x_min), (y - y_min) / (y_max - y_min)) for node, (x, y) in pos.items()}
    G.add_edges_from(edges)

    # Assuming the node id is a tuple (x, y)
    for node in G.nodes():
        nx.set_node_attributes(G, {node: {'x': node[0], 'y': node[1]}})

    # Replace node ids with integer ids starting from 0
    # G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)

    data = nx.node_link_data(G)


    # Extract x coordinates of the nodes
    x_coords = [data['pos'][0] for node, data in G.nodes(data=True)]
    x_array = np.array(x_coords).reshape(-1, 1)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=0).fit(x_array)

    # Get the cluster labels
    labels = kmeans.labels_

    # Get the indices of the nodes in the middle cluster
    middle_cluster_indices = np.where(labels == 1)[0]  # Assuming the middle cluster is labeled as 1

    # Get the x coordinates of the nodes in the middle cluster
    middle_cluster_x_coords = x_array[middle_cluster_indices]

    # Calculate the difference between the largest and smallest x coordinate in the middle cluster
    difference = np.max(middle_cluster_x_coords) - np.min(middle_cluster_x_coords)




    # Create a list of colors for each node
    colors = ['red' if data['pos'][0] in middle_cluster_x_coords else 'black' for node, data in G.nodes(data=True)]

    # Draw the graph
    nx.draw(G, node_color=colors)
    plt.show()

    # plt.figure(figsize=(3, 9))
    # nx.draw(G, pos, node_color='lightblue', edge_color='gray', node_size=25, font_size=10)
    # plt.title('Graph Visualization')
    # plt.show()

    for node_data in data['nodes']:
        for key, value in node_data.items():
            if isinstance(value, np.integer):
                node_data[key] = int(value)
            elif isinstance(value, np.floating):
                node_data[key] = float(value)
            elif isinstance(value, np.ndarray):
                node_data[key] = value.tolist()

    return data


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


# Process multiple images
# image_paths = [str(p)+'.png' for p in range(1,10)]

# graph_data_list = []
# for img_path in image_paths:
#     graph_data = process_image('./maple/'+img_path)
#     graph_data_list.append(graph_data)

process_image('./trees/Maples/1.png')

# Save the serialized graphs to a file
# with open('maples.json', 'w') as f:
#     json.dump(graph_data_list, f, cls=NumpyEncoder)
