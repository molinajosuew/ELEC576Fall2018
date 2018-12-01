from Entity import Entity
from DeepNeuralNetwork import DeepNeuralNetwork
from math import atan, pi, sqrt


class Creature(Entity):
    def __init__(self, x, y, kind, brain_shape):
        super().__init__(x, y, kind)
        self.brain = DeepNeuralNetwork(architecture = brain_shape)
        self.prey = None
        self.predator = None
        self.fitness = 1

    def target_prey_and_predator(self, prey, predator):
        self.prey = prey
        self.predator = predator

    def get_angle_and_distance(self, entity):
        if entity is None:
            return [[0], [0]]

        opposite = self.y - entity.y
        adjacent = entity.x - self.x

        distance = sqrt(opposite ** 2 + adjacent ** 2)

        if adjacent == 0:
            if opposite >= 0:
                return [[.25], [distance]]
            else:
                return [[.75], [distance]]

        angle = atan(opposite / adjacent)

        if adjacent < 0 <= opposite:
            angle = pi + angle
        elif adjacent < 0 and opposite <= 0:
            angle = pi + angle
        elif adjacent > 0 >= opposite:
            angle = 2 * pi + angle

        return [[angle / (2 * pi)], [distance]]

    def move(self, width, height):
        prey_angle = self.get_angle_and_distance(self.prey)[0]
        direction = self.brain.predict([prey_angle] + self.get_angle_and_distance(self.predator))

        if direction == 0:
            self.x = (self.x + 1) % (width + 1)
        elif direction == 1:
            self.x = (self.x + 1) % (width + 1)
            self.y = (self.y + 1) % (height + 1)
        elif direction == 2:
            self.y = (self.y + 1) % (height + 1)
        elif direction == 3:
            self.x = (self.x - 1) % (width + 1)
            self.y = (self.y + 1) % (height + 1)
        elif direction == 4:
            self.x = (self.x - 1) % (width + 1)
        elif direction == 5:
            self.x = (self.x - 1) % (width + 1)
            self.y = (self.y - 1) % (height + 1)
        elif direction == 6:
            self.y = (self.y - 1) % (height + 1)
        else:
            self.x = (self.x + 1) % (width + 1)
            self.y = (self.y - 1) % (height + 1)
