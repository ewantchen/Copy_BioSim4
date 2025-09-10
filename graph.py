import igraph as ig 
import json 
import os
from src.gene import Gene


# On récupère toutes les données de la génération.
def load_generation_data(gen_number):
    folder = os.path.join(os.path.dirname(__file__),"src", "generations")
    with open(os.path.join(folder, f"gen_{gen_number}.json"), "r") as f:
        return json.load(f)


# On crée notre graphe, en récupérant d'abord les données de nos agents et en les 
# stockants. Ensuite, on les transforme en graphe que l'on pourra ensuite afficher.
def create_graph(gen_data, frame_index = 0, agent_id = 1):
    frame = gen_data[frame_index]
    agents = frame["agents"]
    agent_data = agents[str(agent_id)]

    edges = []
    vertices = set() # on utilise un set() pour qu'il n'y ai pas de doublons automatiquement.
    weights = []

    genome = Gene.hex_to_genome(agent_data["genome"])

    for gene in genome:
        source_key = ("N" if gene.sourceType == 0 else "S") + str(gene.sourceNum)
        target_key = ("N" if gene.targetType == 0 else "A") + str(gene.targetNum)

        vertices.add(source_key)
        vertices.add(target_key)
        edges.append((source_key, target_key))
        weights.append(gene.weight)



    print(len(vertices))

    # On doit remap les sommets pour les edges soient dans le bon ordre, 
    # pour que Igraph puisse comprendre.
    vertex_list = sorted(vertices)
    vertex_map = {v: i for i, v in enumerate(vertex_list)}

    mapped_edges = [(vertex_map[src], vertex_map[trgt]) for src, trgt in edges]


    # ces fonctions traitent ensuite des informations stockées pour en faire un graphe
    g = ig.Graph()
    g.add_vertices(len(vertex_list))
    g.add_edges(mapped_edges)


    for gene in genome:     
        # Source
        if gene.sourceType == 0:  # Neurone
            key = f"N{gene.sourceNum}"
        else:  # Sensor
            key = f"S{gene.sourceNum}"
        vertex_index = vertex_map[key]
        g.vs[vertex_index]["name"] = key

        # Target
        if gene.targetType == 0:  # Neurone
            key = f"N{gene.targetNum}"
        else:  # Action
            key = f"A{gene.targetNum}"
        vertex_index = vertex_map[key]
        g.vs[vertex_index]["name"] = key

        vertex_index = vertex_map[key]
        g.vs[vertex_index]["name"] = key

    print(f"Nombre total de gènes: {len(genome)}")
    
    # Analyser chaque gène
    for i, gene in enumerate(genome):
        source_type = "SENSOR" if gene.sourceType == 1 else "NEURON"
        target_type = "ACTION" if gene.targetType == 1 else "NEURON"
        
        print(f"Gène {i}: {source_type}_{gene.sourceNum} -> {target_type}_{gene.targetNum} (weight: {gene.weight})")
        

    g.es["weight"] = weights


    g.vs["id"] = vertex_list

    return g


# imprime le graphe sous forme de .png
def print_graph() :
    data = load_generation_data(2)
    graph = create_graph(data)
    layout = "fruchterman_reingold" # permet d'avoir le style du graphe.
    ig.plot(graph, "graph.png", edge_curved=True, bbox=(400,400), margin=64, layout=layout, vertex_label=graph.vs["name"])

print_graph()
