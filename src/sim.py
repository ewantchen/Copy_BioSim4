from env_goal import BioSim
import time
import pygame
import os
from params import PARAMS

# param_env1 = Params(size = 128, n_agents = 21, max_time= 100, render_mode = "human")
# env1 = BioSim(param_env1)

env = BioSim(size=PARAMS["SIZE"], n_agents=PARAMS["N_AGENTS"], max_time=PARAMS["MAX_TIME"])

saved_generations = [0,1,2,3,4,5,6,7,8,9]
env.reset()
for i in range(PARAMS["NUM_GENERATION"]+1):
    generation_state = [] # Contient toutes les informations de la gen
    print(i)
    for j in range(env.max_time) :
        env.step()
        if i in saved_generations : 
            generation_state.append(env.save_frame_state())

    stats = env.end_of_sim()
    env.save_generation_state(i, stats)

    generation_state.append(stats)
        
    if i in saved_generations :
        env.save_generation_state(i, generation_state)



