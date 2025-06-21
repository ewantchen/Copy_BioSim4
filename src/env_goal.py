# Il y a encore du travail, mais cela indique le but à atteindre pour
# utiliser les instances des objets Agent

import pygame.gfxdraw
from brain import (
    Neuron,
    Gene,
    NeuralNet,
    ACTIONS,
    SENSORS,
    sensor_values
)

from agent import *

from params import PARAMS

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
        "render_modes": ["human", "rgb_array"],
        "render_fps": 120
    }

    def __init__(self, size=PARAMS["SIZE"], n_agents=PARAMS["N_AGENTS"], max_time=100, render_mode=None):
        super().__init__()
        self.n_agents = n_agents

        self.position_occupancy = np.zeros((size, size), dtype=bool)

        self.agents = Agent.all_agents

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
        for i in range(self.n_agents):
            agent = Agent()
            agent.id = i

        if self.render_mode == "human":
            self._render_frame()

        self.timestep = 0
        # pas besoin des observations au début
        observations = {
            agent.get_observation()
            for agent in self.agents
        }
        return observations

    def condition(self):
        # on décrit une condition de séléction selon les besoins. Sera ensuite appelé à la fin
        # de la simulation
        for agent in self.agents:
            x, y = agent.position
            if x > self.size // 2:
                self.rewards[agent] = 0
            else:
                self.rewards[agent] = 1

    def selection(self):
        self.survivors = [agent for agent in self.survivors
                          # condition à revoir
                          if agent in agent.genome]
        new_population = []
        new_genome = {}
        # prendre les agents qui on survécu
        for agent in self.agents:
            if agent in self.survivors:
                # on selection les parents au hasard.
                # Peut être changé dans le futur pour correspondre
                # à la géographie

                parent1 = random.choice(self.survivors)
                parent2 = random.choice(self.survivors)
                # parent1_genome = parent1.gennome
                parent1_genome = agent.genome[parent1]
                parent2_genome = agent.genome[parent2]

                # on prend le genome des parents
                half_genome2 = len(parent2_genome) // 2
                half_genome1 = len(parent1_genome) // 2

                # child = Agent()
                child_genome = parent1_genome[:half_genome1] + parent2_genome[half_genome2:]

                # création du génome des enfants
                # child.genome
                child_genome = Gene.apply_point_mutations(child_genome)
                child_genome = Gene.random_insert_deletion(child_genome)

                new_agent_name = f"agent_{len(new_population)}"
                new_population.append(new_agent_name)
                new_genome[new_agent_name] = child_genome

        # on redifini les agents

        self.agents = new_population
        agent.genome = new_genome
        agent.brain = {
            agent: NeuralNet.create_wiring_from_genome(agent.genome)
            for agent in self.agents
        }
        agent.colors = agent.get_all_colors()

        # on recrée la position de la nouvelle popoulation
        agent.position = agent.new_positions()

    # fonction à appeller lors de la fin d'une simulation, et préparation de la prochaine
    def end_of_sim(self):

        self.condition()

        for agent in self.agents:
            if self.rewards[agent] == 0:
                self.termination[agent] = True
                self.truncations[agent] = True
                self.dead_agents.append(agent)
            elif self.rewards[agent] == 1:
                self.survivors.append(agent)
        for dead in self.dead_agents:
            self.agents.remove(dead)
            del dead.position
            del dead.genome
            del dead.brains
            dead.clear()
        self.selection()

        self.timestep = 0
        observations = {
            agent.get_observation(agent)
            for agent in self.agents
        }
        return observations

    def step(self, actions):
        # On initialise les informations à chaque step pour ensuite utiliser ces informations
        # et les envoyer au prochain step. Rewards permet de sélectionner à la fin d'une sim.
        # Termination et truncations permettent d'éliminer tout les agents non sélectionnés.
        # Infos est une conditions necéssaire pour PettingZoo.
        self.rewards = {agent: 0 for agent in self.agents}
        self.termination = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        infos = {agents: {} for agents in self.agents}

        for agent in self.agents:
            agent.update_and_move()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return MultiDiscrete(np.array([self.size, self.size]))

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(len(ACTIONS))

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        agent = Agent()
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

        """"
        for agent in self.agents :
            color = agent.colors
            pygame.draw.circle(
                canvas,
                color,
                #le + 0.5 permet de centrer le cercle
                (np.array(self.agent_position[agent]) + 0.5) * pix_square_size,
                #rayon du cercle
                pix_square_size / 2,
            )
        """

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


self = BioSim()
self.reset()
print(self.get_observation)
