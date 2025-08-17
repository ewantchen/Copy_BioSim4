# On y met toutes les variables pour éviter de chercher
# dans le code les valeurs à changer.

from survival_condition import *

PARAMS = {
    "SIZE" : 128,
    "FPS"  : 10,
    "NUM_GENERATION" : 10,
    "MAX_TIME" : 100, # Il faut laisser assez de temps pour que les agents puissent atteindre l'objectif
    "N_AGENTS" : 2,
    "MAX_NEURONS" : 50, # Nombre de neurones internes
    "GENOME_LENGTH" : 200,
    "MUTATIONS" : False,
    "SEXUAL_REPRODUCTION" : False,
    "SURVIVAL_CRITERIA" : kill_half_map
}

def condition(self):
         #on décrit une condition de séléction selon les besoins. Sera ensuite appelé à la fin 
         # de la simulation
     for agent in self.agents :
        x, y = agent.position
        if x > self.size // 2:
                self.rewards[agent] = 0
        else : 
            self.rewards[agent] = 1



class Params:
    def __init__(self, size):
        self.size = size