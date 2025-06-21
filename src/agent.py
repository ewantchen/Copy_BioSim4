from brain import (
    Neuron,
    Gene,
    NeuralNet,
    ACTIONS,
    SENSORS,
    sensor_values
)

from params import PARAMS



class Agent:
    size = PARAMS["SIZE"]
    # attribut de classe. Permet d'avoir une propriété commune à tous les agents.
    occupancy = np.zeros((size,size), dtype=bool)
    all_agents = []

    def __init__(self):
        # On détermine l'inex de l'agent lors de l'iniitialisation de l'environnement.
        self.id : int 

        # Les informations propres à l'agents sont sa position, son code génétique,
        # le cerveau qui en découle ainsi que sa couleur. Elles sont toutes déterminés par
        # des fonctions.
        self.position = self.new_position()
        self.genome = Gene.make_random_genome()
        self.brain = NeuralNet.create_wiring_from_genome(self.genome)
        self.color = self.make_genetic_color_value(self.genome)
        self.get_observation()

        # On ajoute à une liste tout les agents à chaque fois qu'ils sont initialisés
        Agent.all_agents.append(self)

        # Fonction qui permet de transformer une action en probabilité
    def Prob2Bool(self, factor: float) -> bool:
        return random.random() < factor

    def __repr__(self):
        return f"Agent('{self.id}', '{self.position}', {self.genome}', '{self.brain}', '{self.color}')"

    def new_position(self):
        while True:
            # Ici, je vous suggère de faire plus simple
            x, y = np.array([
                random.randint(0, self.size - 1),
                random.randint(0, self.size - 1)
            ])
            if not self.occupancy[x, y]:
                self.position = np.array([x, y], dtype=np.int32)
                self.occupancy[x, y] = True
                break

    def update_and_move(self):
        x, y = self.position
        movex = 0.0
        movey = 0.0
        actionLevels = self.brain.feed_forward((x, y), self.size)
        movex += actionLevels[2]
        movex -= actionLevels[3]
        movey += actionLevels[0]
        movey -= actionLevels[1]

        movex = np.tanh(movex)
        movey = np.tanh(movey)

        probX = 1 if self.Prob2Bool(abs(movex)) else 0
        probY = 1 if self.Prob2Bool(abs(movey)) else 0

        signX = -1 if movex < 0.0 else 1
        signY = -1 if movey < 0.0 else 1

        offset = int(probX * signX), int(probY * signY)

        current_x, current_y = self.position
        new_x = current_x + offset[0]
        new_y = current_y + offset[1]

        # ici on vérifie que les nouvelles positions respectent les bordures et la position
        # des autres entités. On change les états de l'ancienne et de la nouvelle position à
        # des états représentatifs du changement.
        if (new_x, new_y) != (current_x, current_y):
            if 0 <= new_x < self.size and 0 <= new_y < self.size:
                if not self.occupancy[new_x, new_y]:
                    self.occupancy[current_x, current_y] = False
                    self.occupancy[new_x, new_y] = True
                    self.position = np.array([new_x, new_y], dtype=np.int32)

    def get_observation(self):
        """Retourne l'observation de l'agent"""
        x, y = self.position
        sensor_values = self.brain._get_sensor_values(
            self.position,
            self.size
        )
        self.observation = {
            'position': [x, y],
            'sensors': sensor_values,
            'neurons': [neuron.output for neuron in self.brain.neurons]
        }


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
                (len(genome) % 2)  # taille du génome modulo 2 en tant que 1er bit
                | ((genome[0].sourceType % 2) << 1)  # on décale le bit d'après
                | ((genome[-1].sourceType % 2) << 2)  # le >> signifie qu'on le place à la suite
                | ((genome[0].targetType % 2) << 3)  #d'un nombre n
                | ((genome[-1].targetType % 2) << 4)
                | ((genome[0].sourceNum % 2) << 5)
                | ((genome[0].targetNum % 2) << 6)
                | ((genome[-1].sourceNum % 2) << 7)
    )

    
    #on transforme les valeurs de couleur en couleur rgb
    # On défini la couleur tel que les valeurs définies par make_genetic_color_value
    # soient isolés à certains bits. Par exemple, le vert est défini par un isolement des 5 
    # premiers bits de c et ces valeurs sont ensuites décalés pour donner une 
    # valeur entre 0 et 255. Le rouge est directement la valeur c
        max_color_val=0xb0
        max_luma_val=0xb0

        r = value
        g = (value & 0x1F) << 3
        b = (value & 0x07) << 5

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
    

