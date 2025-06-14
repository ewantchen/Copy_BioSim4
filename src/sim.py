from env import BioSim
import time
import pygame

env = BioSim(size = 128, n_agents = 20, max_time= 100, render_mode = "human")
observations = env.reset()
running = True

pygame.init()
for generation in range(20) :
    print(f"Génération {generation+1}/20 - Agents : {len(env.agents)}")

    for i in range(env.max_time) :
        observations, rewards, terminations, truncations, infos = env.step({})

    env.end_of_sim()

env.close()
pygame.quit()