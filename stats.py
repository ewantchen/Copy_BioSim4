# Fichier stockant toutes les fonctions permettant de faire des statistiques sur les 
# populations faites. Permet surtout de faire des graphiques.

from src.params import PARAMS
import os
import json

# On récupère toutes les données de la génération.
def load_generation_data(gen_number):
    folder = os.path.join(os.path.dirname(__file__),"src", "generations")
    with open(os.path.join(folder, f"gen_{gen_number}.json"), "r") as f:
        return json.load(f)

# Fonction permettant de calculer le nombre de survivants et
# d'en faire un graphique.
def survival_rate(gen_data) :
    data = gen_data
    dead_num = data["dead_agents"]
    survival_ratio = int(dead_num) / PARAMS["N_AGENTS"]
    return survival_ratio

data = load_generation_data(4)

r = survival_rate(data)
print(r)