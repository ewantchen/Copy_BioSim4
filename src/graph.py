import igraph as ig 
import json 
import os


# On récupère toutes les données de la génération.
def load_generation_data(gen_number):
    folder = os.path.join(os.path.dirname(__file__), "src", "generations")
    with open(os.path.join(folder, f"gen_{gen_number}.json"), "r") as f:
        return json.load(f)


# On crée notre graphe, en récupérant d'abord les données de nos agents et en les 
# stockants. Ensuite, on les transforme en graphe que l'on pourra ensuite afficher.
def create_graph(gen_data, frame_index = 1, agent_id = 1):
    frame = gen_data[frame_index]
    agents = frame["agents"]
    agent_data = agents[str(agent_id)]

    edges = []
    vertices = set() # on utilise un set() pour qu'il n'y ai pas de doublons automatiquement.
    weights = []

    genome = agent_data["genome"]


    for gene in genome : 
        source =gene["sourceNum"]
        target = gene["targetNum"]
        vertices.add(source)
        vertices.add(target)
        edges.append((source, target))
        weights.append(gene["weight"])

        # On doit remap les sommets pour les edges soient dans le bon ordre, 
        # pour que Igraph puisse comprendre.
        vertex_list = sorted(vertices)
        vertex_map = {v: i for i, v in enumerate(vertex_list)}

        mapped_edges = [(vertex_map[src], vertex_map[trgt]) for src, trgt in edges]

        weights.append(gene["weight"])

        # ces fonctions traitent ensuite des informations stockées pour en faire un graphe
        g = ig.Graph()
        g.add_vertices(len(vertex_list))
        g.add_edges(mapped_edges)

        g.es["weight"] = weights


        g.vs["id"] = vertex_list

        return g
    


def print_graph() :
    data = load_generation_data(1)
    graph = create_graph(data)
    layout = 'fruchterman_reingold' # permet d'avoir le style du graphe.
    ig.plot(graph, layout)

print_graph()