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

import pygame
import hashlib

class BioSim(ParallelEnv):
    metadata = {
        "name": "BioSim",
        "render_modes" : ["human", "rgb_array"], 
        "render_fps" : 60
    }

    def __init__(self, size = 128, n_agents = 100, max_time = 100, render_mode=None):
        super().__init__()
        self.n_agents = n_agents
        
        
        self.agent_position = {}
        self.position_occupancy = np.zeros((size,size), dtype=bool)

        self.agent_genome = None

        self.timestep = None
        self.max_time = max_time

        self.render_mode = render_mode
        self.window = None
        self.window_size = 1024
        self.clock = None

        if render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()
            pygame.display.set_mode(
                (self.window_size, self.window_size),
                pygame.RESIZABLE,  
            )

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
                       self.agent_position[agents] = np.array([x,y], dtype=np.int32)
                       self.position_occupancy[x,y] = True
                       positions.add((x,y))
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

        self.agent_colors = self.get_all_colors()       

        if self.render_mode == "human" :
            self._render_frame()

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
              else : 
                  self.rewards[agents] = 1

                   
    
    def selection(self) :
        if not self.survivors : 
            print("aucun n'a survécu")
        self.survivors = [agent for agent in self.survivors
                            if agent in self.agent_genome]
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
        self.agent_colors = self.get_all_colors() 

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
        # On initialise les informations à chaque step pour ensuite utiliser ces informations 
        # et les envoyer au prochain step. Rewards permet de sélectionner à la fin d'une sim.
        # Termination et truncations permettent d'éliminer tout les agents non sélectionnés.
        # Infos est une conditions necéssaire pour PettingZoo. 
        self.rewards = {agents : 0 for agents in self.agents}
        self.termination = {agent: False for agent in self.agents}
        self.truncations = {agents : False for agents in self.agents}
        infos = {agents : {} for agents in self.agents}




        #propagation avant
        for agents in self.agents :
            
            #on prend l'agent et son cerveau
            brain = self.agent_brains[agents]
            if not hasattr(brain, 'neurons') or not brain.neurons:
                continue  # Passe à l'agent suivant si le cerveau est invalide

            # On récupère les valeurs des inputs par la fonction qui calcule les valeurs
            # qui vont de 0.0 à 1.0.
            sensor_values = brain._get_sensor_values(self.agent_position[agents], self.size)
            brain.get_sensors_input(sensor_values)
            brain.feed_forward()

            # Prend en variables les inputs calculés avant et renvoie les résultats calculés. Ça
            # donne des actions qui vont être ensuite sélectionnées.
            action_outputs = brain.get_action_outputs(sensor_values)

            # Action par défaut dans le cas où le programme n'a pas pu trouver une action à envoyer.
            action_name = "STAY"

            # On prend les action avec la plus forte activation.
            if action_outputs :
                best_action_index = max(action_outputs, key = action_outputs.get)
                action_name = ACTIONS[best_action_index]

            # On update les positions des agents comme des variables pour les utiliser après.
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

            # ici on vérifie que les nouvelles positions respectent les bordures et la position
            # des autres entités. On change les états de l'ancienne et de la nouvelle position à
            # des états représentatifs du changement.
            if (new_x, new_y) != (current_x, current_y):
                if not self.position_occupancy[new_x, new_y] :
                    self.position_occupancy[current_x, current_y] = False
                    self.position_occupancy[new_x, new_y] = True
                    self.agent_position[agents] = [new_x, new_y]

            if self.render_mode == "human" :
                self._render_frame()
            
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
        'neurons': [neuron.output for neuron in self.agent_brains[agent].neurons]
    }
        return observation

    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return MultiDiscrete(np.array([self.size, self.size]))

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(len(ACTIONS))
    

    # on transforme la valeur génétique en couleur.
    # Un gène possède 5 informations : sa source, sa cible et son poid
    # ici, on prend le premier et le dernier gène et on transforme par 
    # modulo ces informations en bits. Ensuite, ces bits sont transformés
    # en couleur rgb. 
    def make_genetic_color_value(self, genome):
        # on vérifie s'il y a un génome
        if not genome:
            return 0

        value = (
            (len(genome) % 2) # taille du génome modulo 2 en tant que 1er bit
            | ((genome[0].sourceType % 2) << 1) # on décale le bit d'après
            | ((genome[-1].sourceType % 2) << 2) # le >> signifie qu'on le place à la suite 
            | ((genome[0].sinkType % 2) << 3)#d'un nombre n
            | ((genome[-1].sinkType % 2) << 4)
            | ((genome[0].sourceNum % 2) << 5)
            | ((genome[0].sinkNum % 2) << 6)
            | ((genome[-1].sourceNum % 2) << 7)
    )
        return value
    
    #on transforme les valeurs de couleur en couleur rgb
    # On défini la couleur tel que les valeurs définies par make_genetic_color_value
    # soient isolés à certains bits. Par exemple, le vert est défini par un isolement des 5 
    # premiers bits de c et ces valeurs sont ensuites décalés pour donner une 
    # valeur entre 0 et 255. Le rouge est directement la valeur c
    def genetic_color_to_rgb(self, c, max_color_val=0xb0, max_luma_val=0xb0):
        r = c
        g = (c & 0x1F) << 3
        b = (c & 0x07) << 5

        # Calculer la luminance (même formule que dans le C++)
        luma = (r + r + r + b + g + g + g + g) // 8

        # Réduire les valeurs si elles sont trop claires
        if luma > max_luma_val:
            if r > max_color_val:
                r %= max_color_val
            if g > max_color_val:
                g %= max_color_val
            if b > max_color_val:
                b %= max_color_val

        return (r, g, b)
    
    # Récuère toutes les couleurs des agents et les mets dans un dictionnaire
    def get_all_colors(self):
        self.agent_colors = {}
        for agents in self.agents :
            genome = self.agent_genome[agents]
            value = self.make_genetic_color_value(genome)
            self.agent_colors[agents] = self.genetic_color_to_rgb(value)
        return self.agent_colors
    

    def render(self) : 
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self) : 
        # On met une clock pour garder une trace du temps passé
        if self.clock is None and self.render_mode == "human" :
            self.clock = pygame.time.Clock()
        
        # On crée une surface rempli de blanc
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255,255,255))
        
        
        # taille d'une case en pixels
        pix_square_size = (
            self.window_size / self.size
        ) 

        # Pour chaque agent, on prend son génome, on applique une couleur selon
        # ce génome.
        # On dessine ensuite un cercle selon cette couleur et la position
        for agent in self.agents :
            color = self.agent_colors[agent]
            pygame.draw.circle(
                canvas,
                color,
                #le + 0.5 permet de centrer le cercle
                (np.array(self.agent_position[agent]) + 0.5) * pix_square_size,
                #rayon du cercle
                pix_square_size / 3,
            )

        # on prend la taille de la grille + 1, et on y dessine la grille
        for x in range(self.size + 1) :
         # lignes horizontales
         pygame.draw.line(
            canvas,
            0,
            # Le point de départ est sur l'axe horizontal
            (0, pix_square_size * x),
            # Point d'arrivée est sur la fin de la grille
            (self.window_size, pix_square_size * x),
            width=1,
            )
         # lignes verticales
         pygame.draw.line(
            canvas,
            0,
            (pix_square_size * x, 0),
            (pix_square_size * x, self.window_size),
            width=1,
        )
         

        if self.render_mode == "human":
        # Blit copie le contenu sur la fenêtre
        # Pump gère les évènements internes
        # Display.update affiche la fenêtre
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])

        else:  # rgb_array
            return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )
    
    def close(self) :
        if self.window is not None :
            pygame.display.quit()
            pygame.quit()
            self.window = None
        self.clock = None
