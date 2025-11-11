import igraph as ig
import json
import os
from src.params import *
from src.gene import *
from src.NeuralNet import *

# Chargement des données d'une génération
def load_generation_data(gen_number):
    folder = os.path.join(os.path.dirname(__file__), "src", "generations")
    with open(os.path.join(folder, f"gen_{gen_number}.json"), "r") as f:
        return json.load(f)

# Création du graphe igraph à partir du génome
def create_graph(gen_data, frame_index=1, agent_id=746):
    frame = gen_data["frames"][frame_index]
    agents = frame["agents"]
    agent_data = agents[str(agent_id)]

    # Récupération du génome brut
    raw_genome = hex_to_genome(agent_data["genome"])

    # Construction du cerveau fonctionnel
    functional_brain: NeuralNet = create_wiring_from_genome(raw_genome)
    final_connections = functional_brain.connections

    edges = []
    vertices = set()
    weights = []

    # Parcours des connexions du cerveau
    for gene in final_connections:
        source_key = ("N" if gene.sourceType == 0 else "S") + str(gene.sourceNum)
        target_key = ("N" if gene.targetType == 0 else "A") + str(gene.targetNum)
        vertices.add(source_key)
        vertices.add(target_key)
        edges.append((source_key, target_key))
        weights.append(gene.weight)

    print(f"Génome brut: {len(raw_genome)} gènes. Cerveau fonctionnel: {len(final_connections)} connexions.")
    print(f"Noeuds du cerveau fonctionnel: {sorted(list(vertices))}")

    vertex_list = sorted(list(vertices))
    vertex_map = {v: i for i, v in enumerate(vertex_list)}
    mapped_edges = [(vertex_map[src], vertex_map[trgt]) for src, trgt in edges]

    g = ig.Graph(directed=True)
    g.add_vertices(len(vertex_list))
    g.add_edges(mapped_edges)
    g.vs["name"] = vertex_list
    g.es["weight"] = weights

    return g


# Sauvegarde du graphe en image PNG (avec couleurs)
def save_png_graph(g, path="graph.png", layout_name="fruchterman_reingold"):
    layout = g.layout(layout_name)

    # Définition des couleurs selon le type de nœud
    color_map = {"S": "#C8FACC", "N": "#B8E0FC", "A": "#FFF9A6"}  # mêmes couleurs que Plotly
    node_colors = [color_map.get(v["name"][0], "#B0BEC5") for v in g.vs]  # gris par défaut

    visual_style = {
        "layout": layout,
        "vertex_color": node_colors,
        "vertex_size": 20,
        "vertex_label": g.vs["name"],
        "vertex_label_size": 10,
        "vertex_label_color": "black",
        "edge_width": [1 for _ in g.es],
        "edge_curved": 0.25,
        "bbox": (200, 200),     
        "margin": 20,           
        "background": "white",
    }
    ig.plot(g, path, **visual_style)
    print(f"Graphe sauvegardé dans {path}")


# Exemple d'utilisation
def print_graph_png(gen_index=50, frame_index=1, agent_id=288):
    data = load_generation_data(gen_index)
    graph = create_graph(data, frame_index, agent_id)
    save_png_graph(graph, f"graph{gen_index}.png")

print_graph_png(50)
