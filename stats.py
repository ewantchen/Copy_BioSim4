# Fichier stockant toutes les fonctions permettant de faire des statistiques sur les 
# populations faites. Permet surtout de faire des graphiques.

from src.params import PARAMS
import os
import json
import itertools
import math 
from matplotlib import pyplot as plt

# On récupère toutes les données de la génération.
def load_generation_data(gen_number):
    folder = os.path.join(os.path.dirname(__file__),"src", "generations")
    with open(os.path.join(folder, f"gen_{gen_number}.json"), "r") as f:
        return json.load(f)

# Fonction permettant de calculer le nombre de survivants et
# d'en faire un graphique.
def survival_rate(dead_num) :
    survival_ratio = (PARAMS["N_AGENTS"]-int(dead_num)) / PARAMS["N_AGENTS"]
    return survival_ratio

# Trop compliqué de mesurer la diversité par les genomes, donc on compare les couleurs.
# On place les couleurs dans un cube avec r,g,b comme coordonnées, et on fait la moyenne entre deux
# couleurs dans le cube 3d. Voir documentation pour plus de détails
def genetic_diversity(agents) : 
    colors = [agent.color for agent in agents]
    pop = len(agents)
    total_dist = 0
    for c1, c2 in itertools.combinations(colors, 2):
        total_dist += math.sqrt(sum((a-b)**2 for a,b in zip(c1,c2))) / math.sqrt(255**2*3)
    norm_dist_moy = (2 / (pop*(pop-1))) * total_dist
    return norm_dist_moy

def render_diversity() :
    data_diversity = []
    for i in range(PARAMS["NUM_GENERATION"]) :
        data = load_generation_data(i)
        data_diversity.append(data["genetic_diversity"])
        
    plt.plot(data_diversity)
    plt.ylabel("diversity")
    plt.xlabel("generations")
    plt.show()

def render_survival():
    data_survival = []
    data_point = []
    for i in range(PARAMS["NUM_GENERATION"]) :
        data = load_generation_data(i)
        data_survival.append(data["dead_agents"])
    for deads in data_survival:
        data_point.append(survival_rate(deads))
    plt.plot(data_point)
    plt.ylabel("survival rate")
    plt.xlabel("generations")
    plt.show()

    

    

