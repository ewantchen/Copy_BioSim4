from env import BioSim
import numpy

env = BioSim(size = 128, n_agents = 20, max_time= 100, render_mode = "human")
observations = env.reset()


for generation in range(20) :
    print(f"Génération {generation+1}/20")

    for i in range(100) :
        observations, rewards, terminations, infos = env.step({})
        if hasattr(env, "_render_frame") :
            env._render_frame()
        env.end_of_sim()
    env.close()