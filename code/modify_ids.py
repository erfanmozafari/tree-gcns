import json

# Load the JSON data
with open('maples.json', 'r') as file:
    data = json.load(file)

# Replace vectors with unique numbers
unique_number = 0
vector_to_number = {}  # Dictionary to store mapping of vectors to unique numbers

# Iterate through the JSON data and replace vectors with unique numbers
for graph in data:
    for key in graph:
        for node in graph['nodes']:
            id = tuple(node['id'])
            if not id in vector_to_number:
                vector_to_number[id] = unique_number
                unique_number += 1


for graph in data:
    for key in graph:
        for node in graph['nodes']:
            if isinstance (node['id'],int):
                continue            
            # print(tuple(node['id']))
            id = tuple(node['id'])
            node['id'] = vector_to_number[id]
        for link in graph['links']:
            if isinstance (link['source'],int) or isinstance (link['target'],int):
                continue 
            source = tuple(link['source'])
            target = tuple(link['target'])
            link['source'] = vector_to_number[source]
            link['target'] = vector_to_number[target]


            


# Save the modified JSON data
with open('modified_maples.json', 'w') as file:
    json.dump(data, file, indent=4)