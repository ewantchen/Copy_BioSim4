# Permet de créer le cerveau des agents, et quelles actions ils doivent faire avec la fonction
# feed_forward()


import numpy as np 
import random
from typing import List, Dict, Tuple

from .params import PARAMS
from .gene import *
from .action_sensors import *



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
                actionLevels[gene.targetNum] += inputVal * gene.weight_as_float(gene)
            else :
                neuronAccumulators[gene.targetNum] += inputVal * gene.weight_as_float(gene)
        
        # Retourne actionLevels, représentant les niveau d'activation brute des agents
        return actionLevels
    


    
# On transforme le génome en quelque chose d'utilisable.
def remap_connection_list(genome, net: NeuralNet): 
    # On vide la liste pour être sûr de repartir de zéro
    net.connections = []
    for g in genome : 
        # On crée une COPIE du gène pour ne pas modifier l'original
        gene_copy = Gene()
        gene_copy.sourceType, gene_copy.targetType, gene_copy.weight = g.sourceType, g.targetType, g.weight

        if g.targetType == 0 :
            gene_copy.targetNum = g.targetNum % PARAMS["MAX_NEURONS"]
        else :
            gene_copy.targetNum = g.targetNum % n_ACTIONS

        if g.sourceType == 0 :
            gene_copy.sourceNum = g.sourceNum % PARAMS["MAX_NEURONS"]
        else :
            gene_copy.sourceNum = g.sourceNum % n_SENSORS
            
        net.connections.append(gene_copy)
        
# Node list permet de recenser tous les neurones. On veut savoir
# ses infos (voir Node())
def make_node_list(net : NeuralNet):
    node_map : Dict[int, Node] = {}
    for conn in net.connections : 
        if conn.targetType == 0 :
            if conn.targetNum not in node_map:
                node_map[conn.targetNum] = Node()
            
            if conn.sourceType == 0 and conn.sourceNum == conn.targetNum :
                node_map[conn.targetNum].numSelfInputs += 1
            else : 
                node_map[conn.targetNum].numOtherInputs += 1


        if conn.sourceType == 0 : 
            if conn.sourceNum not in node_map:
                node_map[conn.sourceNum] = Node()

            node_map[conn.sourceNum].numOutputs += 1
            
    return node_map


def remove_neuron(node_map, net: NeuralNet, neuron_num):
    i = 0
    # On utilise while pour ne sauter aucun élément
    while i < len(net.connections):
        conn = net.connections[i]
        # On cherche les connexions qui VONT VERS le neurone à supprimer
        if conn.targetType == 0 and conn.targetNum == neuron_num:
            # Si la source est un autre neurone, on décrémente son compteur de sorties
            if conn.sourceType == 0 and conn.sourceNum in node_map:
                node_map[conn.sourceNum].numOutputs -= 1
            
            # On supprime la connexion
            net.connections.pop(i)
            # On incrémente pas i, car le prochain élément a pris la place de celui-ci
        else:
            # La connexion est valide, on passe à la suivante
            i += 1
            
    if neuron_num in node_map:
        del node_map[neuron_num]



def cull_useless_neurons(node_map, net: NeuralNet):
    all_done = False
    while not all_done:
        all_done = True
        # On itère sur une copie de la liste des clés pour pouvoir supprimer de la map en toute sécurité
        for neuron_num in list(node_map.keys()):
            # Il se peut que le neurone ait déjà été supprimé dans cette même passe (effet cascade)
            if neuron_num not in node_map:
                continue

            node = node_map[neuron_num]
            # Condition d'inutilité : aucune sortie, ou ne s'alimente que lui-même
            if node.numOutputs == node.numSelfInputs:
                all_done = False
                # On nettoie les connexions qui vont vers ce neurone et on le supprime de la map
                remove_neuron(node_map, net, neuron_num)


def create_wiring_from_genome(genome: List[Gene]) -> "NeuralNet":
    net = NeuralNet()

    # Appliquer le modulo et peupler la liste de connexions
    remap_connection_list(genome, net)

    # Créer la carte initiale des neurones et de leurs propriétés
    node_map = make_node_list(net)

    # Élaguer le réseau en supprimant les neurones inutiles en cascade
    cull_useless_neurons(node_map, net)

    # Remapper les neurones restants avec des index séquentiels (0, 1, 2...)
    neuron_remap = {old: new for new, old in enumerate(sorted(node_map.keys()))}

    for gene in net.connections:
        if gene.sourceType == 0:
            gene.sourceNum = neuron_remap[gene.sourceNum]
        if gene.targetType == 0:
            gene.targetNum = neuron_remap[gene.targetNum]

    # Créer la liste finale de neurones pour le réseau
    net.neurons = [Neuron() for _ in range(len(neuron_remap))]
    for old_idx, node in node_map.items():
        if node.numOtherInputs > 0:
            new_idx = neuron_remap[old_idx]
            net.neurons[new_idx].driven = True
    
    # trier les connexions pour l'efficacité de feed_forward
    net.connections.sort(key=lambda g: g.targetType)

    return net
