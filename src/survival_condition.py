# On crée ici les conditions sous forme de fonction que l'on importe.


def kill_half_map(size, agents) :
    # on décrit une condition de séléction selon les besoins. Sera ensuite appelé à la fin
    # de la simulation
    for agent in agents:
        x, y = agent.position
        if size // 2 < x : 
            agent.alive = True
        else :
            agent.alive = False