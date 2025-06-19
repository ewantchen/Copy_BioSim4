from brain import *

class Agent:
    # attribut de classe
    occupancy = np.zeros((size,size), dtype=bool)
    id = 0
    #colors =

    def __init__(self):
        self.id = id
        id += 1
        self.position = self.new_position()
        self.genome = Gene.make_random_genome()
        # self.neural_net =…
        self.brain = NeuralNet.create_wiring_from_genome(self.genome)
        self.color = ""


    def new_position(self):
        while True:
            x, y = np.array([
                random.randint(0, self.size - 1),
                random.randint(0, self.size - 1)
            ])
            if not self.occupancy[x, y]:
                self.position = np.array([x, y], dtype=np.int32)
                self.occupancy[x, y] = True
                break


    def set_color(self, color):
        self.color = color