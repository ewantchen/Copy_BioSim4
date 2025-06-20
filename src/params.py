# On y met toutes les variables pour éviter de chercher
# dans le code les valeurs à changer.

PARAMS = {
    "SIZE" : 128,
    "FPS"  : 0,
    "N_AGENTS" : 1,
    "GENOME_LENGTH" : 1
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