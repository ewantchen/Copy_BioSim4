# Il y a encore du travail, mais cela indique le but à atteindre pour
# utiliser les instances des objets Agent

import pygame.gfxdraw
from NeuralNet import *
from gene import *

from agent import *

from params import PARAMS

import functools
from gymnasium.spaces import Discrete, MultiDiscrete
from pettingzoo import ParallelEnv
import random
import numpy as np

import pygame
import hashlib

import json
import os
import matplotlib


class BioSim(ParallelEnv):
    metadata = {
        "name": "BioSim",
        "render_modes": ["human", "rgb_array"],
        "render_fps": PARAMS["FPS"]
    }

    def __init__(self, size=PARAMS["SIZE"], n_agents=PARAMS["N_AGENTS"], max_time=100, render_mode=None):
        self.n_agents = n_agents

        #self.position_occupancy = np.zeros((size, size), dtype=bool)
        self.agents_map = np.zeros((size, size), dtype=bool)

        self.agents = []

        self.timestep = None
        self.max_time = max_time

        self.render_mode = render_mode
        self.window = None
        self.window_size = 512
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

    def reset(self, seed=None, options=None):
        #Agent.all_agents = []

        for i in range(self.n_agents):
            agent = Agent(self.agents_map)
            agent.id = i
            self.agents.append(agent)

        """"
        if self.render_mode == "human" :
            self.render_frame()
        """

        #self.agents = agent.all_agents

        self.timestep = 0


        # pas besoin des observations au début
        observations = {
            agent.get_observation(self.agents_map)
            for agent in self.agents
        }

        return observations


    def condition(self):
        # on décrit une condition de séléction selon les besoins. Sera ensuite appelé à la fin
        # de la simulation
        for agent in self.agents:
            x, y = agent.position
            if x > self.size // 2:
                agent.alive = False
            else:
                agent.alive = True


    def create_genetic_offsprings(self):
        # on selection les parents au hasard.
        # Peut être changé dans le futur pour correspondre
        # à la géographie

        parent1 = random.choice(self.survivors)
        parent2 = random.choice(self.survivors)


        # On reprend la logique de bioSim4 pour faire une transmission similaire aux allèles. 
        # On fait en sorte que le génome soit celui de l'un des deux parents, et qu'une partie soit 
        # celle de l'autre parent. 


        g1 = parent1.genome
        g2 = parent2.genome

        if PARAMS["SEXUAL_REPRODUCTION"] is True :
            # On prend le génome le plus court des deux parents. Il fera office de génome de référence.
            child_genome = g1 if len(g1) >= len(g2) else g2
            gShorter = g2 if len(g1) >= len(g2) else g1

            # Dans le génome le plus court, on prend un espace, défini aléatoirement,
            # qui remplacera la partie du gène qu'elle couvre par l'espace. On s'assure aussi
            # que les indexs font sens en terme de taille. Par exemple :
            # genome = [A, A, A, A, A, A, A, A, A, A] gShorter = [B, B, B, B, B, B, B]
            #index0 = 2
            #index1 = 5
            #On prend dans gShorter la tranche de l'indice 2 (inclus) à 5 
            # exclu) → éléments 2, 3, 4 → [B, B, B]
            #On copie cette tranche dans genome à partir de l'indice 2
            #Après la copie, genome devient :
            #[A, A, B, B, B, A, A, A, A, A]
            size = len(gShorter)
            index0 = random.randint(0, size - 1)
            index1 = random.randint(0, size) 
            if index0 > index1:
                index0, index1 = index1, index0

            # Notre génome est remplacé dans l'espace entre les indexs
            child_genome[index0:index1] = gShorter[index0:index1]

            # Ici, on fait en sorte que le génome fasse la taille moyenne du génome des parents.
            # On ajoute 1 si la longueur des 2 génomes additionnés est impair.
            total = len(g1) + len(g2)
            if total % 2 == 1 and random.randint(0, 1) == 1:
                total += 1
            new_length = total // 2

            # Si le génome est trop long, on coupe aléatoirement soit le bout du début,
            # soit de la fin.
            """
            if len(child_genome) > new_length :
                to_trim = len(child_genome) - new_length
                if random.random() < 0.5:
                    child_genome = child_genome[to_trim:]
                else:
                    # trim from back
                    child_genome = child_genome[:-to_trim]
            """

            child_genome = Gene.apply_point_mutations(child_genome)
            child_genome = Gene.random_insert_deletion(child_genome)    


            return child_genome
        
        else :
            child_genome = g2
            return child_genome

        
  

   
# Fonction permettant de créer la prochaine génération avec la fonction de création de offsprings.
    def new_population(self):
        self.agents = []
        for i in range(self.n_agents):
            agent = Agent(self.agents_map)
            agent.id = i
            agent.genome = self.create_genetic_offsprings()
            agent.brain = NeuralNet.create_wiring_from_genome(agent.genome)
            agent.color = agent.make_genetic_color_value()
            self.agents.append(agent)


    # fonction à appeller lors de la fin d'une simulation, et préparation de la prochaine,
    # similaire à Reset()
    def end_of_sim(self):

        self.condition()

        for agent in self.agents:
            if not agent.alive :
                self.dead_agents.append(agent)
            else:
                self.survivors.append(agent)

        self.new_population()

        self.dead_agents = []
        self.survivors = []


        self.timestep = 0
        observations = {
            agent.get_observation(self.agents_map)
            for agent in self.agents
        }
        return observations

    def step(self, action):
        # On initialise les informations à chaque step pour ensuite utiliser ces informations
        # et les envoyer au prochain step. Rewards permet de sélectionner à la fin d'une sim.
        # Termination et truncations permettent d'éliminer tout les agents non sélectionnés.
        # Infos est une condition necéssaire pour PettingZoo.
        # Rewards pourra être utiliser plus tard pour faire un score de fitness pour 
        # choisir mieux les parents.
        self.rewards = {agent: 0 for agent in self.agents}
        self.termination = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}


        for agent in self.agents:
            agent.update_and_move(self.agents_map)
            
        """"
        if self.render_mode == "human" :
            self.render_frame()
        """

        observations = {
            agent.get_observation(self.agents_map)
            for agent in self.agents
        }


        return observations, self.rewards, self.termination, self.truncations, infos
    

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return MultiDiscrete(np.array([self.size, self.size]))

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(len(ACTIONS))



    def render_frame(self):
        # On met une clock pour garder une trace du temps passé
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # On crée une surface rempli de blanc
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # taille d'une case en pixels
        pix_square_size = (
                self.window_size / self.size
        )

        # Pour chaque agent, on prend son génome, on applique une couleur selon
        # ce génome.
        # On dessine ensuite un cercle selon cette couleur et la position

    
        for agent in self.agents :
            pygame.draw.circle(
                canvas,
                agent.color,
                #le + 0.5 permet de centrer le cercle
                (np.array(agent.position) + 0.5) * pix_square_size,
                #rayon du cercle
                pix_square_size / 2,
            )


        """"
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
        )"""

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

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
        self.clock = None


    def save_frame_state(self) :
        frame_state = []        
        frame_state.append({
            "frame" : self.timestep,
        })

        for agent in self.agents : 
            frame_state.append({
                "id" : agent.id,
                "alive" : agent.alive,
                "position" : agent.position,
                "color" : agent.color,
                "genome" : [{"soureType" : g.sourceType, 
                             "source": g.sourceNum,
                             "targetType" : g.targetType,
                             "target": g.targetNum, 
                             "weight": g.weight
                             } 
                             for g in agent.genome]
            })

        return frame_state

    def save_generation_state(self, gen_number, generation_state) :


        folder = os.path.join(os.path.dirname(__file__), "generations")

        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, f"gen_{gen_number}.json"), "w") as f:
            json.dump(generation_state, f, indent=2)



    def render_generation(self, gen_number):
        folder = os.path.join(os.path.dirname(__file__), "generations")
        with open(os.path.join(folder, f"gen_{gen_number}.json"), "r") as f:
            generation_state = json.load(f)

        if not pygame.get_init() : 
            pygame.init()

        self.clock = pygame.time.Clock()
        window = pygame.display.set_mode((self.window_size, self.window_size))
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size

        running = True
        frame_index = 0

        while running and frame_index < self.max_time:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            window.fill((255, 255, 255))  # Nettoyer la fenêtre à chaque frame

            frame_state = generation_state[frame_index]
            agents = frame_state[1:]  # Ignorer le premier dict "frame"

            for agent in agents:
                x, y = agent["position"]
                color = agent["color"]
                pygame.draw.circle(
                    window,
                    color,
                    ((x + 0.5) * pix_square_size, (y + 0.5) * pix_square_size),
                    pix_square_size / 2,
                )

            pygame.display.flip()  # Met à jour tout l’écran

            self.clock.tick(self.metadata["render_fps"])  # Contrôle vitesse (fps)

            frame_index += 1

        pygame.quit()

            




