# On y met toutes les variables pour éviter de chercher
# dans le code les valeurs à changer.

#from src.survival_condition import *

PARAMS = {
    "SIZE" : 128,
    "FPS"  : 10,
    "NUM_GENERATION" : 12,
    "MAX_TIME" : 100, # Il faut laisser assez de temps pour que les agents puissent atteindre l'objectif
    "N_AGENTS" : 10,
    "MAX_NEURONS" : 1, # Nombre de neurones internes, minimum 1
    "GENOME_LENGTH" : 10,
    "MUTATIONS" : False,
    "SEXUAL_REPRODUCTION" : False,
    #"SURVIVAL_CRITERIA" : kill_half_map
}



