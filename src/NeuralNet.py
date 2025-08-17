# Permet de créer le cerveau des agents, et quelles actions ils doivent faire avec la fonction
# feed_forward()


import numpy as np 
import random
from typing import List, Dict, Tuple

from params import PARAMS
from gene import Gene
from action_sensors import *



# On défini un objet Node, qui permet d'avoir toutes les informations sur
# chaque neurones. Ça permet de supprimper tout les neurones inutiles.
class Node : 
    def __init__(self):
        self.numOutputs  = 0
        self.numSelfInputs = 0
        self.numOtherInputs = 0

    
        
"""Conversion du génome en réseau neuronal"""
class Neuron :
    # La classe neuron permet de représenter un neurone dans le réseau neuronal d'un agent.
    # Elle gère les signeaux entrants et sortants. On y initialise des valeurs arbitraire
    # destinées à changer. 
    def __init__(self):
        self.output : float = 0.5 # Représente la valeur de sortie des neurones
        self.driven: bool = False # Indique si le neurone reçoit un input. Si c'est False,
        # le neurone ne sera pas activé
        self.input : float = 0.0 # Accumule les inputs avant l'activation. Réinitialisé à
        # 0.0 avant chaque propagation.


class NeuralNet :
    def __init__(self):
        self.connections : List[Gene] = []
        self.neurons : List[Neuron] = []
    

    # Cette fonction permet de trouver les valeurs des sensors qui sont dans le réseau 
    # de neurones.
    def get_sensor_values(self, agent_position, world_size) -> Dict[int, float]:
        x, y = agent_position
    
        return {
                0: sensor_values["X_POS"](x, world_size),
                1: sensor_values["Y_POS"](y, world_size),
                2: sensor_values["RANDOM"](),
                3: sensor_values["BOUNDARY_DIST_X"](x, world_size),
                4: sensor_values["BOUNDARY_DIST_Y"](y, world_size)
                }



    # Cette fonction permet de trouver le résultat du calcul du réseau de neurones
    # Pour ce faire, on initialise des variables. Ensuite, on regarde dans chaque connexions :
    # - Si la destination est une action (targetType == 1) et qu'on n’a pas encore calculé
    # les sorties des neurones, on applique la fonction tanh() sur tous les neurones "driven",
    # pour obtenir leurs outputs dans l'intervalle [-1.0, 1.0],
    # - si la source est un sensor, alors l'input value du neurone est égal à l'output des sensor
    # qui est trouvé par la fonction get_sensor_values. Sinon, la valeur de l'input est égale
    # à l'output du neurone source. 
    # - si la cible est une action, alors on ajoute l'output du neurone à actionLevels, sinon on l'ajoute
    # à neuronAccumulators.
    def feed_forward(self, agent_position, world_size) :
        
        actionLevels = [0.0] * n_ACTIONS  # Tableau pour dire quelle valeurs sont les plus fortes.
        neuronAccumulators = [0.0] * len(self.neurons) # Tableau pour trier les outputs de chaque neurones.
        
        neuronOutputsComputed = False # Permet de calculer la valeur de tout les neurones internes avant.
        sensor_vals = self.get_sensor_values(agent_position, world_size) # On récupère les valeurs des sensors dès le début.
        
        for gene in self.connections :
            # Cette condition est utilisé à la fin du code.
            if gene.targetType == 1 and neuronOutputsComputed == False :
                for i, neuron in enumerate(self.neurons) :
                    if neuron.driven == True :
                        neuron.output = np.tanh(neuronAccumulators[i])
                neuronOutputsComputed = True

            # Les deux conditions d'après sont générales.
            if gene.sourceType == 1 :
                inputVal = sensor_vals[gene.sourceNum]
            else :
                inputVal = self.neurons[gene.sourceNum].output
            
            if gene.targetType == 1 :
                actionLevels[gene.targetNum] += inputVal * gene.weight_as_float()
            else :
                neuronAccumulators[gene.targetNum] += inputVal * gene.weight_as_float()
        
        # Retourne actionLevels, représentant les niveau d'activation brute des agents
        return actionLevels
    


    
    # Permet de faire des liens entre les neurones du génome. Liste les neurones et
    # les connections entre eux.
    @staticmethod
    def create_wiring_from_genome(genome: List[Gene], max_neurons=1000) -> "NeuralNet" :
            # L'objet NeuralNet est décri comme une liste de genes(connexions) et une 
            # liste de neurones
            net = NeuralNet()       

            # On initialise une liste. On prend toutes les connexions et on ajoute 
            # l'index de leurs cibles et leurs sources. Si c'est un sensor ou 
            # une action, alors l'index dépend de la liste d'actions/sensors
            # tandis que l'index du neurone interne dépend du nombre de neurones
            # max
            net.connections = []
            # On ajoute les gene du génome dans NeuralNet, dans net.connections
            for gene in genome : 
                if gene.targetType == 0 :
                    gene.targetNum %= PARAMS["MAX_NEURONS"]
                else :
                    gene.targetNum %= n_ACTIONS

                if gene.sourceType == 0 :
                    gene.sourceNum %= PARAMS["MAX_NEURONS"]
                else :
                    gene.sourceNum %= n_SENSORS
                

                # Ces nouvelles connexions sont ensuite ajoutées dans une liste "connections"
                net.connections.append(gene)
                
            # On crée une liste avec tout les neurones internes du génome.
            # On n'ajoute pas les neurones actions et sensors.
            # Pour chaque connexion, on regarde sa source et sa cible. On les décrits ensuite
            # dans node_map comme un node. Voir la class Node. Node_map permet de voir quels connexions 
            # sont inutiles et les supprimer. Permet aussi ensuite de faire la liste de neurones
            # pour le neural net.
            node_map : dict[int, Node] = {}
            for gene in net.connections : 
                if gene.targetType == 0 :
                    if gene.targetNum not in node_map and gene.targetNum < 0x7FFF :
                        node_map[gene.targetNum] = Node()

                    # On ajoute aussi aux variables les infos qu'on a.
                    if gene.sourceType == 0 and gene.sourceNum == gene.targetNum :
                        node_map[gene.targetNum].numSelfInputs += 1
                    else : 
                        node_map[gene.targetNum].numOtherInputs += 1


                if gene.sourceType == 0 : 
                    if gene.sourceNum not in node_map and gene.sourceNum < 0x7FFF :
                        node_map[gene.sourceNum] = Node()
                        node_map[gene.sourceNum].numOutputs += 1
                

            

            # On ajoute un moyen de trier et supprimer tous les neurones inutiles.
            # Tant que l'on supprime des nodes, on recommence la boucle.
            # On utilise ensuite node_map comme une liste, et dans cette liste, chaque 
            # fois que la condition d'inutilité est vérifiée pour un neurone, on regarde dans notre liste 
            # net.connections si des connexions vont vers ce neurone.
            # Si c’est le cas, on les supprime une par une. Et si la source de ces connexions
            # est un autre neurone, on décrémente son nombre de sorties (numOutputs).
            # Une fois toutes les connexions vers ce neurone supprimées, on supprime
            # le neurone lui-même de node_map.
            # On recommence tant que des suppressions sont effectuées, car cela peut
            # rendre d'autres neurones inutiles à leur tour (effet en cascade).
            alldone = False
            while alldone == False:
                alldone = True
                for neuron in list(node_map):
                    node = node_map[neuron]
                    if node.numOutputs == 0 or node.numOutputs == node.numSelfInputs :
                        i = 0
                        removed_connections = 0
                        while i < len(net.connections):
                            gene = net.connections[i]
                            # Si le connexion concerne un neurone interne et qu'il est dans la liste de neurones
                            if (gene.targetType == 0 and gene.targetNum == neuron) or (gene.sourceType == 0 and gene.sourceNum == neuron):
                                # Si la connexion supprime la sortie d'un autre neurone, décrémente le compteur
                                if gene.sourceType == 0 and gene.sourceNum in node_map:
                                    node_map[gene.sourceNum].numOutputs -= 1
                                if gene.targetType == 0 and gene.targetNum in node_map:
                                    node_map[gene.targetNum].numOtherInputs -= 1
                                net.connections.pop(i)
                                removed_connections += 1
                            else:
                                i += 1
                        del node_map[neuron]
                        alldone = False



            # Tout les neurones provenant du génome sont classés de façon 
            # numérotés. Ça permet à la fonction Feedfoward() de parcourir le réseau
            # dans l'ordre.
            neuron_remap = {old: new for new, old in enumerate(sorted(node_map))}


            
            # On indexe aussi les connexions selon la position des neurones avec neuron_remap.
            # Ça permet de mieux retrouver nos connexions plus tard.
            for gene in net.connections:
                if gene.targetType == 0:
                    gene.targetNum = neuron_remap[gene.targetNum]
                if gene.sourceType == 0:
                    gene.sourceNum = neuron_remap[gene.sourceNum]
            

            # On ajoute tous les neurones à NeuralNet. Ça donne une liste de neurones
            # Voir l'objet NeuralNet()
            net.neurons = [Neuron() for _ in range(len(neuron_remap))]
            

            # On retourne les deux listes
            return net
      