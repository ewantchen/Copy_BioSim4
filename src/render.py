from env_goal import BioSim

from generations import *

env = BioSim()

for i in range(10):
    env.render_generation(i)
