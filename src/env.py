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
from typing import Optional


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

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Affiche la simulation avec Pygame.
        - Les agents sont des cercles colorés selon leur génome
        - Fond de grille discret
        
        Args:
            mode: 'human' (affichage) ou 'rgb_array' (retourne image)
        Returns:
            Array numpy si mode='rgb_array', sinon None
        """
        if not hasattr(self, '_render_initialized'):
            self._init_render()

        # Création surface
        surface = pygame.Surface((self.screen_size, self.screen_size))
        surface.fill(self.colors['background'])
        
        # Dessin de la grille
        self._draw_grid(surface)
        
        # Dessin des agents (billes colorées)
        self._draw_agents_as_spheres(surface)
        
        # Infos texte
        self._draw_simulation_info(surface)
        
        if mode == 'human':
            pygame.event.pump()
            self.screen.blit(surface, (0, 0))
            pygame.display.flip()
        elif mode == 'rgb_array':
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(surface)), 
                axes=(1, 0, 2))
        return None

    def _init_render(self):
        """Initialise Pygame et les paramètres visuels"""
        pygame.init()
        self.cell_size = 20  # Taille d'une cellule en pixels
        self.screen_size = self.size * self.cell_size
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("BioSim - Visualisation Évolutive")
        
        # Configuration des couleurs
        self.colors = {
            'background': (255, 255, 255),  # Blanc
            'grid': (230, 230, 230),        # Gris très clair
            'text': (0, 0, 0)               # Noir
        }
        
        self.font = pygame.font.SysFont('Arial', 14)
        self._render_initialized = True

    def _draw_grid(self, surface):
        """Dessine la grille de fond"""
        for x in range(0, self.screen_size, self.cell_size):
            pygame.draw.line(
                surface, self.colors['grid'], 
                (x, 0), (x, self.screen_size), 1)
        for y in range(0, self.screen_size, self.cell_size):
            pygame.draw.line(
                surface, self.colors['grid'], 
                (0, y), (self.screen_size, y), 1)

    def _draw_agents_as_spheres(self, surface):
        """Dessine les agents comme des cercles colorés selon leur génome"""
        for agent_id, pos in self.agent_position.items():
            # Conversion du génome en couleur hex -> RGB
            genome = self.agent_genome[agent_id]
            color = self._genome_to_color(genome)
            
            # Position et taille de la bille
            center = (
                int(pos[0] * self.cell_size + self.cell_size / 2),
                int(pos[1] * self.cell_size + self.cell_size / 2)
            )
            radius = int(self.cell_size * 0.4)  # 80% de la cellule
            
            # Dessin avec effet 3D simple
            pygame.draw.circle(
                surface, 
                color, 
                center, 
                radius
            )
            # Bordure pour effet relief
            pygame.draw.circle(
                surface,
                (min(255, color[0] + 30), 
                (center[0] - 2, center[1] - 2), 
                radius, 
                2)
            )

    def _genome_to_color(self, genome: str) -> tuple:
        """
        Convertit un génome hexadécimal en couleur RGB.
        Ex: "A3F" -> (163, 255, 63)
        """
        # Normalisation à 6 caractères (doublage si besoin)
        hex_str = (genome * 6)[:6]
        
        try:
            # Conversion hex -> RGB
            return (
                int(hex_str[0:2], 16) % 256,
                int(hex_str[2:4], 16) % 256,
                int(hex_str[4:6], 16) % 256
            )
        except:
            return (100, 100, 100)  # Couleur par défaut si erreur

    def _draw_simulation_info(self, surface):
        """Affiche les informations textuelles"""
        info = [
            f"Génération: {self.generation}",
            f"Agents: {len(self.agent_position)}",
            f"Temps: {self.timestep}/{self.max_time}"
        ]
        
        for i, text in enumerate(info):
            text_surface = self.font.render(text, True, self.colors['text'])
            surface.blit(text_surface, (10, 10 + i * 20))
        
