# On crée ici les conditions sous forme de fonction que l'on importe.
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def kill_half_map(size, agents) :
    safe_place = 20, 100
    # on décrit une condition de séléction selon les besoins. Sera ensuite appelé à la fin
    # de la simulation
    for agent in agents:
        x, y = agent.position
        agent.alive = x/size