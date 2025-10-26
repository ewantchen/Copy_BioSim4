# On y met toutes les variables pour éviter de chercher
# dans le code les valeurs à changer.

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.survival_condition import *

PARAMS = {
    "SIZE" : 128,
    "FPS"  : 30 ,    
    "WINDOW_SIZE" : 512,
    "NUM_GENERATION" : 101,
    "MAX_TIME" : 300, # Il faut laisser assez de temps pour que les agents puissent atteindre l'objectif
    "N_AGENTS" : 1024,
    "MAX_NEURONS" : 2, # Nombre de neurones internes, minimum 1
    "GENOME_LENGTH" : 8,
    "MUTATIONS" : False,
    "SEXUAL_REPRODUCTION" : True,
    "SURVIVAL_CRITERIA" : kill_half_map,

}

PARAMS["FPS"] = PARAMS["MAX_TIME"] / 10


