import random

ACTIONS = [
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
]

n_ACTIONS = len(ACTIONS)

SENSORS = [
   "X_POS",
   "Y_POS",
   "RANDOM",
   "BOUNDARY_DIST_X",
   "BOUNDARY_DIST_Y",
]

n_SENSORS = len(SENSORS)

sensor_values = {
    # Position X normalisée [0, 1] où 0 = bord gauche, 1 = bord droit
    "X_POS": lambda agent_x, world_size: float(agent_x) / float((world_size - 1)),
    
    # Position Y normalisée [0, 1] où 0 = bord bas, 1 = bord haut
    "Y_POS": lambda agent_y, world_size: float(agent_y) / float((world_size - 1)),
    
    # Valeur aléatoire pour l'aléatoire [0, 1]
    "RANDOM": lambda *_: random.random(),
    
    # Distance au bord X normalisée [0, 1] où 0 = au bord, 1 = au centre
    "BOUNDARY_DIST_X": lambda agent_x, world_size: min(float(agent_x), 
                    float(world_size) - 1 - float(agent_x)) / float((world_size) // 2),
    
    # Distance au bord Y normalisée [0, 1] où 0 = au bord, 1 = au centre
    "BOUNDARY_DIST_Y": lambda agent_y, world_size: min(float(agent_y),
            float(world_size) - 1 - float(agent_y)) / float((world_size) // 2)
}