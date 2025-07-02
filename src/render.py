from env_goal import BioSim



from igraph import *
import matplotlib.pyplot as plt 


import time 


env = BioSim()

env.render_generation(0)
env.render_generation(100)
