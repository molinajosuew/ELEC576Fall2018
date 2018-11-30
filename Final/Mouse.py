from Entity import Entity
from DeepNeuralNetwork import DeepNeuralNetwork
from math import atan, pi, sqrt


class Mouse(Entity):
    def __init__(self, x = 0, y = 0):
        super().__init__(x = x, y = y, kind = 'mouse')
        self.brain = DeepNeuralNetwork()
        self.prey = None
        self.fitness = 1

    def target_prey(self, prey):
        self.prey = prey

    def get_prey_angle(self):
        if self.prey is None:
            return 0

        opposite = self.y - self.prey.y
        adjacent = self.prey.x - self.x

        if adjacent == 0:
            return 0

        angle = atan(opposite / adjacent)

        if adjacent < 0 <= opposite:
            angle = pi + angle
        elif adjacent < 0 and opposite <= 0:
            angle = pi + angle
        elif adjacent > 0 >= opposite:
            angle = 2 * pi + angle

        angle /= 2 * pi

        return angle

    def move(self, width, height):
        direction = self.brain.predict([[self.get_prey_angle()]])

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
