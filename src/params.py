# On y met toutes les variables pour éviter de chercher
# dans le code les valeurs à changer.

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.survival_condition import *

PARAMS = {
    "SIZE" : 100,
    "FPS"  : 30 ,    
    "WINDOW_SIZE" : 1024,
    "NUM_GENERATION" : 1200,
    "MAX_TIME" : 300, # Il faut laisser assez de temps pour que les agents puissent atteindre l'objectif
    "N_AGENTS" : 50,
    "MAX_NEURONS" : 2, # Nombre de neurones internes, minimum 1
    "GENOME_LENGTH" : 6,
    "MUTATIONS" : True,
    "SEXUAL_REPRODUCTION" : True,
    "SURVIVAL_CRITERIA" : kill_half_map,

}

PARAMS["FPS"] = PARAMS["MAX_TIME"] / 10


