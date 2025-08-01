from env_goal import BioSim
import time
import pygame
import os
from params import PARAMS

# param_env1 = Params(size = 128, n_agents = 21, max_time= 100, render_mode = "human")
# env1 = BioSim(param_env1)

env = BioSim(size=PARAMS["SIZE"], n_agents=PARAMS["N_AGENTS"], max_time=PARAMS["MAX_TIME"])
""""
running = True
generation = 0
observations = env.reset()
pygame.init()

for i in range(200) : 
    print(f"generation {i}")
    for i in range(env.max_time) :    
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if not running :
            exit()
        observations, rewards, terminations, truncations, infos = env.step({})

    env.end_of_sim()



env.close()
pygame.quit()
"""

observations = env.reset()
saved_generations = [0,1,2,3,4,5,6,7,8,9]
for i in range(PARAMS["NUM_GENERATION"]+1):
    generation_state = []
    print(i)
    for j in range(env.max_time) :
        observations, rewards, terminations, truncations, infos = env.step({})
        if i in saved_generations : 
            generation_state.append(env.save_frame_state())
    env.end_of_sim()
        
    if i in saved_generations :
        env.save_generation_state(i, generation_state)



