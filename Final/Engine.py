import os

from Entity import Entity
from Creature import Creature
from random import randint, random
from math import sqrt, ceil
from numpy import array


class Engine(object):
    def __init__(self, world_width, world_height, n_mice, n_cheese, n_cats, n_generations, cross_over_rate, mutation_rate, file_name = None):
        self.world_width = world_width
        self.world_height = world_height
        self.n_mice = n_mice
        self.n_cheese = n_cheese
        self.n_cats = n_cats
        self.n_generations = n_generations
        self.cross_over_rate = cross_over_rate
        self.mutation_rate = mutation_rate

        self.capture_radius = { 'mouse': 20, 'cat': 30 }
        self.generation_life = 10000
        self.generation_time = 0
        self.brain_shape = [3, 16, 16, 16, 8]

        self.entities = []

        for i in range(0, n_cheese):
            self.entities.append(Entity(x = randint(0, self.world_width), y = randint(0, self.world_height), kind = 'cheese'))

        if file_name is None:
            for i in range(0, n_mice):
                self.entities.append(Creature(x = randint(0, self.world_width), y = randint(0, self.world_height), kind = 'mouse', brain_shape = self.brain_shape))

            for i in range(0, n_cats):
                self.entities.append(Creature(x = randint(0, self.world_width), y = randint(0, self.world_height), kind = 'cat', brain_shape = self.brain_shape))
        else:
            file = open(file_name, "r")
            lines = file.readlines()
            file.close()

            for i in range(0, len(lines), 3):
                entity = Creature(x = randint(0, world_width), y = randint(0, world_height), kind = lines[i].replace('\n', ''), brain_shape = self.brain_shape)
                entity.brain.W = eval(lines[i + 1])
                entity.brain.b = eval(lines[i + 2])
                self.entities.append(entity)

        for entity in self.entities:
            if entity.kind == 'mouse' or entity.kind == 'cat':
                self.assign_prey_and_predator(entity)

    def get_fittest(self, kind):
        fittest = None

        for entity in self.entities:
            if entity.kind == kind and (fittest is None or entity.fitness > fittest.fitness):
                fittest = entity

        return fittest

    def tick(self):
        if self.generation_time == self.generation_life:
            mice = []

            for entity in self.entities:
                if entity.kind == 'mouse':
                    mice.append(entity)

            cats = []

            for entity in self.entities:
                if entity.kind == 'cat':
                    cats.append(entity)

            # Elitism

            best_mouse = self.get_fittest('mouse')
            self.entities.remove(best_mouse)
            second_best_mouse = self.get_fittest('mouse')

            best_cat = self.get_fittest('cat')
            self.entities.remove(best_cat)
            second_best_cat = self.get_fittest('cat')

            self.entities = [best_mouse, second_best_mouse, best_cat, second_best_cat]

            for i in range(0, self.n_mice - 2):
                self.entities.append(self.create_child(self.roulette_wheel(mice), self.roulette_wheel(mice), 'mouse'))

            for i in range(0, self.n_cheese):
                self.entities.append(Entity(x = randint(0, self.world_width), y = randint(0, self.world_height), kind = 'cheese'))

            for i in range(0, self.n_cats - 2):
                self.entities.append(self.create_child(self.roulette_wheel(cats), self.roulette_wheel(cats), 'cat'))

            self.generation_time = 0

            for entity in self.entities:
                if entity.kind == 'mouse' or entity.kind == 'cat':
                    self.assign_prey_and_predator(entity)

            self.save_generation("mice_generation")

            return False
        else:
            self.generation_time += 1

            for entity in self.entities:
                if entity.kind == 'mouse' or entity.kind == 'cat':
                    self.assign_fitness(entity)

            for entity in self.entities:
                if entity.kind == 'mouse' or entity.kind == 'cat':
                    entity.move(width = self.world_width, height = self.world_height)

            for entity in self.entities:
                if entity.kind == 'mouse' or entity.kind == 'cat':
                    self.assign_prey_and_predator(entity)

            return True

    @staticmethod
    def roulette_wheel(creatures):
        total_fitness = 0

        for candidate in creatures:
            total_fitness += max(1, candidate.fitness)

        ball = random()
        wheel = 0

        for candidate in creatures:
            wheel += max(1, candidate.fitness) / total_fitness

            if wheel > ball:
                return candidate

    def create_child(self, dominant_gene, recessive_gene, kind):
        child = Creature(x = randint(0, self.world_width), y = randint(0, self.world_height), kind = kind, brain_shape = self.brain_shape)

        # Cross-Over
        for key in child.brain.W:
            for i in range(0, len(child.brain.W[key])):
                for j in range(0, len(child.brain.W[key][i])):
                    if random() < self.cross_over_rate:
                        child.brain.W[key][i, j] = dominant_gene.brain.W[key][i, j]
                    else:
                        child.brain.W[key][i, j] = recessive_gene.brain.W[key][i, j]
        for key in child.brain.b:
            for i in range(0, len(child.brain.b[key])):
                if random() < self.cross_over_rate:
                    child.brain.b[key][i] = dominant_gene.brain.b[key][i]
                else:
                    child.brain.b[key][i] = recessive_gene.brain.b[key][i]

        # Mutation
        for key in child.brain.W:
            for i in range(0, len(child.brain.W[key])):
                for j in range(0, len(child.brain.W[key][i])):
                    if random() < self.mutation_rate:
                        child.brain.W[key][i, j] = (child.brain.B - child.brain.A) * random() + child.brain.A
                    else:
                        child.brain.W[key][i, j] = (child.brain.B - child.brain.A) * random() + child.brain.A
        for key in child.brain.b:
            for i in range(0, len(child.brain.b[key])):
                if random() < self.mutation_rate:
                    child.brain.b[key][i] = (child.brain.B - child.brain.A) * random() + child.brain.A
                else:
                    child.brain.b[key][i] = (child.brain.B - child.brain.A) * random() + child.brain.A

        return child

    def re_spawn(self, entity):
        entity.x = randint(0, self.world_width)
        entity.y = randint(0, self.world_height)

    @staticmethod
    def distance(entity1, entity2):
        return sqrt((entity1.x - entity2.x) ** 2 + (entity1.y - entity2.y) ** 2)

    def assign_fitness(self, creature):
        if self.distance(creature, creature.prey) < self.capture_radius[creature.kind]:
            creature.fitness += 100
            self.re_spawn(creature.prey)

            if creature.prey.kind == 'mouse':
                creature.prey.fitness -= 100

    def assign_prey_and_predator(self, creature):
        prey = None
        predator = None

        for entity in self.entities:
            if creature.kind == 'mouse' and entity.kind == 'cheese' or creature.kind == 'cat' and entity.kind == 'mouse':
                if prey is None or self.distance(creature, entity) < self.distance(creature, prey):
                    prey = entity
            if creature.kind == 'mouse' and entity.kind == 'cat':
                if predator is None or self.distance(creature, entity) < self.distance(creature, predator):
                    predator = entity

        creature.target_prey_and_predator(prey, predator)

    def save_generation(self, file_name):
        file = open(file_name + '_backup.txt', "w")
        lines = []

        for entity in self.entities:
            if entity.kind != 'cheese':
                lines.append(entity.kind + '\n')
                lines.append(str(entity.brain.W).replace('\n', '').replace(' ', '') + '\n')
                lines.append(str(entity.brain.b).replace('\n', '').replace(' ', '') + '\n')

        file.writelines(lines)
        file.close()
        os.rename(file_name + '_backup.txt', file_name + '.txt')
