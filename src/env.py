from brain import (
    Neuron,
    Gene,
    NeuralNet,
    ACTIONS,
    sensor_values
)
import functools
from gymnasium.spaces import Discrete, MultiDiscrete
from pettingzoo import ParallelEnv
import random
import numpy as np


class BioSim(ParallelEnv):
    metadata = {
        "name": "BioSim",
    }

    def __init__(self, size = 128, n_agents = 10, max_time = 100):
        self.n_agents = n_agents
        
        
        self.agent_position = {}
        self.position_occupancy = np.zeros((size,size), dtype=bool)

        self.agent_genome = None

        self.timestep = None
        self.max_time = max_time
        self.size = size
        self.survivors = []
        self.dead_agents = []

    def new_positions(self):
        positions = set()
        for agents in self.agents :
             while True :
                  x, y = np.array([
                random.randint(0, self.size - 1),
                random.randint(0, self.size - 1)
                ])
                  if not self.position_occupancy[x,y] :
                       self.agent_position[agents] = np.array([x,y], dtype=np.int(32))
                       self.position_occupancy[x,y] = True
                       positions.add(np.array([x,y]))
                       break
        return self.agent_position
         

    def reset(self, seed=None, options=None):
        #on prend les agents de la liste
        self.agents = [f"agent_{i}" for i in range(self.n_agents)]
        #on leur donne une position aléatoire 
        
        self.agent_position = self.new_positions()

        self.agent_genome = {
            agents : Gene.make_random_genome()
            for agents in self.agents
        }
        self.agent_brains = {
            agent: NeuralNet.create_wiring_from_genome(self.agent_genome[agent])
            for agent in self.agents
            }


        self.timestep = 0
        #pas besoin des observations au début
        observations = {
            agents : self._get_observation(agents)
            for agents in self.agents
        }
        return observations
    
    def condition(self):
         #on décrit une condition de séléction selon les besoins. Sera ensuite appelé à la fin 
         # de la simulation
         for agents in self.agents :
              x, y = self.agent_position[agents]
              if self.timestep >= self.max_time and x > self.size // 2:
                   self.rewards[agents] = 0

                   
    
    def selection(self, agents) :
         
         new_population = []
         new_genome = {}
         #prendre les agents qui on survécu
         for agents in self.agents :
              if agents in self.survivors :
                   #on selection les parents au hasard.
                   #Peut être changé dans le futur pour correspondre
                   #à la géographie
                   parent = random.choice(self.survivors)
                   parent_genome = self.agent_genome[parent]
                   
                   #on prend le genome des parents
                   half_genome = len(parent_genome) // 2
                   child_genome = parent_genome[:half_genome]

                    #création du génome des enfants
                   child_genome = Gene.apply_point_mutations(child_genome)
                   child_genome = Gene.random_insert_deletion(child_genome)
                   
                   new_agent_name = f"agent_{len(new_population)}"
                   new_population.append(new_agent_name)
                   new_genome[new_agent_name] = child_genome

        # on redifini les agents
         self.agents = new_population
         self.agent_genome = new_genome
         self.agent_brains = {
            agent: NeuralNet.create_wiring_from_genome(self.agent_genome[agent])
            for agent in self.agents
            }

        #on recrée la position de la nouvelle popoulation
         self.agent_position = self.new_positions()

    #fonction à appeller lors de la fin d'une simulation, et préparation de la prochaine               
    def end_of_sim(self) :

        self.condition()

        for agents in self.agents :
             if self.rewards[agents] == 0 :
                  self.termination[agents] = True
                  self.truncations[agents] = True
                  self.dead_agents.append(agents)
             elif self.rewards[agents] == 1 :
                  self.survivors.append(agents)
        for dead in self.dead_agents :
             self.agents.remove(dead)
             del self.agent_position[dead]
             del self.agent_genome[dead]
             del self.agent_brains[dead]
        self.dead_agents.clear()
        self.selection()

        self.timestep = 0
        observations = {
            agents : self._get_observation(agents)
            for agents in self.agents
        }
        return observations
        

             
         

    def step(self, actions):
        self.rewards = {agents : 0 for agents in self.agents}
        self.termination = {agent: False for agent in self.agents}
        self.truncations = {agents : False for agents in self.agents}
        infos = {agents : {} for agents in self.agents}

        #propagation avant
        for agents in self.agents :
            
            #on prend l'agent et son cerveau
            brain = self.agent_brains[agents]

            #on prend les valeurs des capteurs
            #on les met dans le réseau
            sensor_values = brain._get_sensor_values(self.agent_position[agents], self.size)
            brain.get_sensors_input(sensor_values)
            #on fait la propagation avant
            brain.feed_forward()
            action_outputs = brain.get_action_outputs()

            if action_outputs :
                best_action_index = max(action_outputs, key = action_outputs.get)
                action_name = ACTIONS[best_action_index]


        current_x, current_y = self.agent_position[agents]
        new_x, new_y = current_x, current_y

        #le max et le min permet d'avoir des bordures en comparant 0 et position
        # Le 0, 0 commence en haut à gauche, donc y doit être rapproché de 0 pour monter.
        if action_name == "NORTH":
            new_y = max(0, current_y - 1)
        elif action_name == "SOUTH":
            new_y = min(self.size - 1, current_y + 1)
        elif action_name == "WEST":
            new_x = max(0, current_x - 1)
        elif action_name == "EAST":
            new_x = min(self.size - 1, current_x + 1)
        elif action_name == "NORTH WEST":
            new_x = max(0, current_x - 1)
            new_y = max(0, current_y - 1)
        elif action_name == "NORTH EAST":
            new_x = min(self.size - 1, current_x + 1)
            new_y = max(0, current_y - 1)
        elif action_name == "SOUTH WEST":
            new_x = max(0, current_x - 1)
            new_y = min(self.size - 1, current_y + 1)
        elif action_name == "SOUTH EAST":
            new_x = min(self.size - 1, current_x + 1)
            new_y = min(self.size - 1, current_y + 1)
        # Déplace l'agent vers le bas

        if (new_x, new_y) != (current_x, current_y):
            if not self.position_occupancy[new_x, new_y] :
                self.position_occupancy[current_x, current_y] = False
                self.position_occupancy[new_x, new_y] = True
                self.agent_position[agents] = [new_x, new_y]
        
        
        self.timestep += 1 
        
        observations = {
            agents : self._get_observation(agents)
            for agents in self.agents
        }
        return observations, self.rewards, self.termination, self.truncations, infos
    
    def _get_observation(self, agent):
        """Retourne l'observation de l'agent"""
        x, y = self.agent_position[agent]
        sensor_values = self.agent_brains[agent]._get_sensor_values(
            self.agent_position[agent],
            self.size
        )
        observation = {
        'position': [x, y],
        'sensors': sensor_values,
        'neurons': [Neuron.output for neuron in self.agent_brains[agent].neurons]
    }
        return observation

    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return MultiDiscrete(np.array([self.size, self.size]))

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(len(ACTIONS))