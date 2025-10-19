import igraph as ig
import json
import os
from src.params import *
from src.gene import *
from src.NeuralNet import *
import plotly.graph_objects as go
from plotly.offline import plot

def load_generation_data(gen_number):
    folder = os.path.join(os.path.dirname(__file__), "src", "generations")
    with open(os.path.join(folder, f"gen_{gen_number}.json"), "r") as f:
        return json.load(f)


# On crée notre graphe, en récupérant d'abord les données de nos agents et en les
# stockants. Ensuite, on les transforme en graphe que l'on pourra ensuite afficher.
def create_graph(gen_data, frame_index=1, agent_id=13):
    frame = gen_data["frames"][frame_index]
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
    g.vs["id"] = vertex_list  # ou g.vs["label"] = vertex_list pour l'affichage

    return g

def save_html_graph(g, path="graph.html", layout_name="fruchterman_reingold", with_arrows=True):
    L = g.layout(layout_name)
    xs = [L[i][0] for i in range(g.vcount())]
    ys = [L[i][1] for i in range(g.vcount())]

    # Arêtes
    edge_x, edge_y = [], []
    for u, v in g.get_edgelist():
        edge_x += [xs[u], xs[v], None]
        edge_y += [ys[u], ys[v], None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                            line=dict(width=1), hoverinfo="none")

    # Nœuds + labels
    labels = g.vs["name"] if "name" in g.vs.attributes() else list(map(str, range(g.vcount())))
    node_trace = go.Scatter(x=xs, y=ys, mode="markers+text",
                            text=labels, textposition="top center",
                            marker=dict(size=10), hoverinfo="text")

    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(showlegend=False,
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      margin=dict(l=20, r=20, t=20, b=20),
                      plot_bgcolor="white")

    # Flèches optionnelles si graphe orienté
    if with_arrows and g.is_directed():
        fig.update_layout(annotations=[
            dict(ax=xs[u], ay=ys[u], x=xs[v], y=ys[v],
                 xref="x", yref="y", axref="x", ayref="y",
                 showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=1)
            for u, v in g.get_edgelist()
        ])

    plot(fig, filename=path, auto_open=False, include_plotlyjs=True)

# Exemple d’usage
def print_graph_html(gen_index):
    data = load_generation_data(gen_index)
    graph = create_graph(data)          # votre fonction existante
    save_html_graph(graph, "graph"+str(gen_index)+".html", with_arrows=True)

print_graph_html(0)