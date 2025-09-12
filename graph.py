import igraph as ig 
import json 
import os
from src.params import *
from src.gene import *
from src.NeuralNet import *


# On récupère toutes les données de la génération.
def load_generation_data(gen_number):
    folder = os.path.join(os.path.dirname(__file__),"src", "generations")
    with open(os.path.join(folder, f"gen_{gen_number}.json"), "r") as f:
        return json.load(f)


# On crée notre graphe, en récupérant d'abord les données de nos agents et en les 
# stockants. Ensuite, on les transforme en graphe que l'on pourra ensuite afficher.
def create_graph(gen_data, frame_index = 0, agent_id = 4):
    frame = gen_data[frame_index]
    agents = frame["agents"]
    agent_data = agents[str(agent_id)]

    # On récupère le génome brut (l'ADN)
    raw_genome = hex_to_genome(agent_data["genome"])
    
    # On construit le cerveau fonctionnel à partir du génome
    # Cette fonction va faire le modulo, l'élagage (culling) et le remappage.
    functional_brain: NeuralNet = create_wiring_from_genome(raw_genome)

    # On travaille maintenant avec les connexions du cerveau final, pas le génome brut
    final_connections = functional_brain.connections

    edges = []
    vertices = set()
    weights = []

    # On parcourt les connexions du cerveau ÉLAGUÉ et REMAPPÉ
    for gene in final_connections:
        # Les numéros sont maintenant petits et séquentiels (0, 1, 2...)
        source_key = ("N" if gene.sourceType == 0 else "S") + str(gene.sourceNum)
        target_key = ("N" if gene.targetType == 0 else "A") + str(gene.targetNum)

        vertices.add(source_key)
        vertices.add(target_key)
        edges.append((source_key, target_key))
        weights.append(gene.weight)

    print(f"Génome brut: {len(raw_genome)} gènes. Cerveau fonctionnel: {len(final_connections)} connexions.")
    print(f"Noeuds du cerveau fonctionnel: {sorted(list(vertices))}")

    # Le reste de votre code pour créer le graphe avec igraph reste identique...
    vertex_list = sorted(list(vertices))
    vertex_map = {v: i for i, v in enumerate(vertex_list)}
    mapped_edges = [(vertex_map[src], vertex_map[trgt]) for src, trgt in edges]

    g = ig.Graph()
    g.add_vertices(len(vertex_list))
    g.add_edges(mapped_edges)

    # et le reste de la fonction pour assigner les noms, etc.
    g.vs["name"] = vertex_list
    g.es["weight"] = weights
    g.vs["id"] = vertex_list # ou g.vs["label"] = vertex_list pour l'affichage

    return g


# imprime le graphe sous forme de .png
def print_graph() :
    data = load_generation_data(2)
    graph = create_graph(data)
    layout = "fruchterman_reingold" # permet d'avoir le style du graphe.
    ig.plot(graph, "graph.png", edge_curved=True, bbox=(400,400), margin=64, layout=layout, vertex_label=graph.vs["name"])

print_graph()
