from env_goal import BioSim
import time
import pygame

from params import PARAMS

# param_env1 = Params(size = 128, n_agents = 21, max_time= 100, render_mode = "human")
# env1 = BioSim(param_env1)

env = BioSim(size=PARAMS["SIZE"], n_agents=PARAMS["N_AGENTS"], max_time=100, render_mode="human")

running = True
generation = 0
env.reset()
pygame.init()
while generation < 10000 and running :
    print(f"Génération {generation+1}/20 - Agents : {len(env.agents)}")



    for i in range(env.max_time) :    
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if not running :
            break
        observations, rewards, terminations, truncations, infos = env.step({})


    env.end_of_sim()
    generation += 1

env.close()
pygame.quit()