# Code permettant de gérer les connexions entre les neurones. Permet de créer le génome.

import numpy as np 
import random
from typing import List, Dict, Tuple
from params import PARAMS

from action_sensors import *


# Un Gene est défini comme une connexion entre deux neurones. Il comporte comme information
# son type de source, l'index de la source, son type de de cible, l'index de la cibe et 
# enfin, le poid de la connexion.
class Gene : 
    def __init__(self):
        self.sourceType: int = 0 #0=NEURON, 1=SENSOR
        self.sourceNum: int = 0 # Index de la source (d'où vient l'input)
        self.targetType: int = 0 #0=NEURON, 1=ACTION
        self.targetNum: int = 0 #Index de la cible (où va l'output)
        self.weight: int = 0 # Poids (int16)


    def weight_as_float(self) -> float : 
        #Converti le poids entier en float [-1.0 , 1.0]
        return self.weight / 8192.0 #Même méthode dans BioSim4
    

    # On défini au tout début comme un weight comme une plage énorme de donnée pour pouvoir
    # mieux faire muter les poids. Quand on l'applique au feedforward on le met en float, dans 
    # une plage entre -4.0 et 4.0
    def make_random_weight(self) -> float :
        #Poid aléatoire (comme dans BioSim4)
        return np.random.randint(-32768, 32767) # int16 signé


    def make_random_gene(self) -> "Gene" :
        #crée un gène (comme dans BioSim4)
        gene = Gene()
        gene.sourceType = np.random.randint(0,2) # 0=NEURON, 1=SENSOR
        gene.sourceNum = np.random.randint(0, 0x7FFF) # 15 bits (comme BioSim)
        gene.targetType = np.random.randint(0, 2) # 0=NEURON, 1=ACTION
        gene.targetNum = np.random.randint(0, 0x7FFF) # 15 bits
        gene.weight = self.make_random_weight()
        return gene
    

    def make_random_genome(self, min_len=10, max_len=50) -> "List[Gene]" :
        #Crée un génome aléatoire
        length = PARAMS["GENOME_LENGTH"]
        return [self.make_random_gene() for _ in range(length)]
    
    # On change avec une chance de 20% à chaque fois de changer par un bit l'information d'un individu.
    # Cette méthode marche très bien en C++, qui a un controle sur tout les bits.
    def random_bit_flip(self, gene : "Gene") -> "Gene" :
        chance = np.random.rand()
        if chance < 0.2 :
            self.sourceType ^= 1 #Flip entre 0 et 1
        elif chance < 0.4 :
            self.targetType ^= 1
        elif chance < 0.6 :
            self.sourceNum ^= (1 << np.random.randint(0,15))
        elif chance < 0.8 :
            self.targetNum ^= (1 << random.randint(0, n_ACTIONS))
        else :
           self.weight ^= (1 << np.random.randint(0,15))
        return gene
    
    def apply_point_mutations(genome : List["Gene"], mutation_rate = 0.01) -> List["Gene"] :
        #applique des mutations aléatoires (comme BioSim)
        for gene in genome : 
            if np.random.rand() < mutation_rate : 
                Gene.random_bit_flip(gene)
        return genome

    
    def random_insert_deletion(genome: List["Gene"], max_length=100) -> List["Gene"] :
        #ajoute ou supprime un gène aléatoirement
        if np.random.rand() < 0.05 : # 5% de chance 
            if np.random.rand() < 0.5 and len(genome) > 1 :
                genome.pop(np.random.randint(0,len(genome))) #supprime
            elif len(genome) < max_length :
                genome.append(Gene.make_random_gene())
        return genome