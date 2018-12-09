import os

from Entity import Entity
from Creature import Creature
from random import randint, random
from math import sqrt
from numpy import array


class Engine(object):
    def __init__(self, world_width, world_height, n_mice, n_cheese, n_cats, n_generations, cross_over_rate, mutation_rate, chromosomes_file = None, stats_file = None):
        self.world_width = world_width
        self.world_height = world_height
        self.n_mice = n_mice
        self.n_cheese = n_cheese
        self.n_cats = n_cats
        self.n_generations = n_generations
        self.cross_over_rate = cross_over_rate
        self.mutation_rate = mutation_rate
        self.chromosomes_file = chromosomes_file
        self.stats_file = stats_file

        self.capture_radius = { 'mouse': 20, 'cat': 30 }
        self.generation_life = 10000
        self.generation_time = 0
        self.brain_shape = [2, 32, 8]

        self.entities = []

        for i in range(0, n_cheese):
            self.entities.append(Entity(x = randint(0, self.world_width), y = randint(0, self.world_height), kind = 'cheese'))

        try:
            file = open('./data/' + chromosomes_file + '.txt', 'r')
            lines = file.readlines()
            file.close()

            print('loading chromosomes')

            for i in range(0, len(lines), 3):
                entity = Creature(x = randint(0, world_width), y = randint(0, world_height), kind = lines[i].replace('\n', ''), brain_shape = self.brain_shape)
                entity.brain.W = eval(lines[i + 1])
                entity.brain.b = eval(lines[i + 2])
                self.entities.append(entity)
        except IOError:
            print('creating chromosomes')

            for i in range(0, n_mice):
                self.entities.append(Creature(x = randint(0, self.world_width), y = randint(0, self.world_height), kind = 'mouse', brain_shape = self.brain_shape))

            for i in range(0, n_cats):
                self.entities.append(Creature(x = randint(0, self.world_width), y = randint(0, self.world_height), kind = 'cat', brain_shape = self.brain_shape))

        for entity in self.entities:
            if entity.kind == 'mouse' or entity.kind == 'cat':
                self.assign_prey_and_predator(entity)

    def get_fittest(self, kind):
        fittest = None

        for entity in self.entities:
            if entity.kind == kind and (fittest is None or entity.fitness > fittest.fitness):
                fittest = entity

        return fittest

    def sort_generation(self):
        sorted_generations = { }

        for entity in self.entities:
            if entity.kind != 'cheese':
                if entity.kind in sorted_generations:
                    sorted_generations[entity.kind].append(entity)
                else:
                    sorted_generations[entity.kind] = []

        for creature in sorted_generations:
            sorted_generations[creature].sort(key = lambda x: x.fitness, reverse = True)

        return sorted_generations

    def tick(self):
        if self.generation_time == self.generation_life:
            self.generation_time = 0
            self.save_stats()

            sorted_generations = self.sort_generation()
            self.entities = []

            for key in sorted_generations:
                for i in range(0, 2):
                    sorted_generations[key][i].fitness = 1
                    self.entities.append(sorted_generations[key][i])

            for i in range(0, self.n_mice - 2):
                self.entities.append(self.create_child(self.roulette_wheel(sorted_generations['mouse']), self.roulette_wheel(sorted_generations['mouse'])))

            for i in range(0, self.n_cheese):
                self.entities.append(Entity(x = randint(0, self.world_width), y = randint(0, self.world_height), kind = 'cheese'))

            for i in range(0, self.n_cats - 2):
                self.entities.append(self.create_child(self.roulette_wheel(sorted_generations['cat']), self.roulette_wheel(sorted_generations['cat'])))

            for entity in self.entities:
                if entity.kind == 'mouse' or entity.kind == 'cat':
                    self.assign_prey_and_predator(entity)

            self.save_chromosomes()

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

    def create_child(self, parent_1, parent_2):
        child = Creature(x = randint(0, self.world_width), y = randint(0, self.world_height), kind = parent_1.kind, brain_shape = self.brain_shape)

        # Cross-Over

        for key in child.brain.W:
            for i in range(0, len(child.brain.W[key])):
                for j in range(0, len(child.brain.W[key][i])):
                    if random() < self.cross_over_rate:
                        child.brain.W[key][i, j] = parent_1.brain.W[key][i, j]
                    else:
                        child.brain.W[key][i, j] = parent_2.brain.W[key][i, j]

        for key in child.brain.b:
            for i in range(0, len(child.brain.b[key])):
                if random() < self.cross_over_rate:
                    child.brain.b[key][i] = parent_1.brain.b[key][i]
                else:
                    child.brain.b[key][i] = parent_2.brain.b[key][i]

        # Mutation

        for key in child.brain.W:
            for i in range(0, len(child.brain.W[key])):
                for j in range(0, len(child.brain.W[key][i])):
                    if random() < self.mutation_rate:
                        child.brain.W[key][i, j] = (child.brain.B - child.brain.A) * random() + child.brain.A

        for key in child.brain.b:
            for i in range(0, len(child.brain.b[key])):
                if random() < self.mutation_rate:
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

    def save_chromosomes(self):
        file = open('./data/' + self.chromosomes_file + '_backup.txt', 'w')
        lines = []

        for entity in self.entities:
            if entity.kind != 'cheese':
                lines.append(entity.kind + '\n')
                lines.append(str(entity.brain.W).replace('\n', '').replace(' ', '') + '\n')
                lines.append(str(entity.brain.b).replace('\n', '').replace(' ', '') + '\n')

        file.writelines(lines)
        file.close()

        try:
            os.remove('./data/' + self.chromosomes_file + '.txt')
        except IOError:
            pass

        os.rename('./data/' + self.chromosomes_file + '_backup.txt', './data/' + self.chromosomes_file + '.txt')

    def save_stats(self):
        file = open('./data/' + self.stats_file + '.txt', 'a')
        file.write('mouse')

        for entity in self.entities:
            if entity.kind == 'mouse':
                file.write(str(entity.fitness) + ',')

        file.write('\n')
        file.write('cat')

        for entity in self.entities:
            if entity.kind == 'cat':
                file.write(str(entity.fitness) + ',')

        file.write('\n')
        file.close()
