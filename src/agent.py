from brain import *
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


        # On ajoute à une liste tout les agents à chaque fois qu'ils sont initialisés
        Agent.all_agents.append(self)

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
                # Ici aussi, vous héritez de la lourdeur du code en C
                # self.position = x, y
                self.position = np.array([x, y], dtype=np.int32)
                self.occupancy[x, y] = True
                break



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
    

