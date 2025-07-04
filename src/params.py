# On y met toutes les variables pour éviter de chercher
# dans le code les valeurs à changer.

PARAMS = {
    "SIZE" : 128,
    "FPS"  : 30,
    "MAX_TIME" : 300, # Il faut laisser assez de temps pour que les agents puissent atteindre l'objectif
    "N_AGENTS" : 100,
    "MAX_NEURONS" : 1,
    "GENOME_LENGTH" : 4,
    "SEXUAL_REPRODUCTION" : False
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