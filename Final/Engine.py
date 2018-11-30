import os

from Entity import Entity
from Mouse import Mouse
from random import randint, random
from math import sqrt
from numpy import array


class Engine(object):
    def __init__(self, world_width, world_height, n_mice, n_cheese, n_generations, cross_over_rate, mutation_rate, file_name = None):
        self.world_width = world_width
        self.world_height = world_height
        self.n_mice = n_mice
        self.n_generations = n_generations
        self.cross_over_rate = cross_over_rate
        self.mutation_rate = mutation_rate

        self.capture_radius = 20
        self.generation_life = 10000
        self.generation_time = 0

        mice = []

        if file_name is None:
            for i in range(0, n_mice):
                mice.append(Mouse(x = randint(0, world_width), y = randint(0, world_height)))
        else:
            file = open(file_name, "r")
            lines = file.readlines()
            file.close()

            for i in range(0, len(lines), 2):
                mouse = Mouse(x = randint(0, world_width), y = randint(0, world_height))
                mouse.brain.w = eval(lines[i])
                mouse.brain.b = eval(lines[i + 1])
                mice.append(mouse)

        cheese = []

        for i in range(0, n_cheese):
            cheese.append(Entity(x = randint(0, world_width), y = randint(0, world_height)))

        self.entities = []
        self.entities.append(mice)
        self.entities.append(cheese)

        for mouse in mice:
            self.assign_prey(mouse)

    def reset(self):
        entities = [sub_entity for sub_entities in self.entities for sub_entity in sub_entities]

        for entity in entities:
            self.re_spawn(entity)

            if entity.kind == 'mouse':
                entity.fitness = 1
                self.assign_prey(entity)

    def get_best_mouse(self):
        best_mouse = None

        for mouse in self.entities[0]:
            if best_mouse is None or mouse.fitness > best_mouse.fitness:
                best_mouse = mouse

        return best_mouse

    def tick(self):
        if self.generation_time == self.generation_life:
            # Elitism
            best_mouse = self.get_best_mouse()
            self.entities[0].remove(best_mouse)
            second_best_mouse = self.get_best_mouse()
            new_mice_generation = [best_mouse, second_best_mouse]

            while len(new_mice_generation) < self.n_mice:
                dominant_gene = self.roulette_wheel()
                recessive_gene = self.roulette_wheel()
                new_mice_generation.append(self.create_child(dominant_gene, recessive_gene))

            self.entities[0] = new_mice_generation
            self.generation_time = 0
            self.reset()

            self.save_generation("mice_generation.txt")

            return False
        else:
            self.generation_time += 1

            for creature in self.entities[0]:
                self.assign_fitness(creature)

            for creature in self.entities[0]:
                creature.move(width = self.world_width, height = self.world_height)

            for creature in self.entities[0]:
                self.assign_prey(creature)

            return True

    def roulette_wheel(self):
        total_fitness = 0

        for mouse in self.entities[0]:
            total_fitness += mouse.fitness

        ball = random()
        wheel = 0

        for mouse in self.entities[0]:
            wheel += mouse.fitness / total_fitness

            if wheel > ball:
                return mouse

        return None

    def create_child(self, dominant_gene, recessive_gene):
        child = Mouse(x = randint(0, self.world_width), y = randint(0, self.world_height))

        # Cross-Over
        for key in child.brain.w:
            for i in range(0, len(child.brain.w[key])):
                for j in range(0, len(child.brain.w[key][i])):
                    if random() < self.cross_over_rate:
                        child.brain.w[key][i, j] = dominant_gene.brain.w[key][i, j]
                    else:
                        child.brain.w[key][i, j] = recessive_gene.brain.w[key][i, j]
        for key in child.brain.b:
            for i in range(0, len(child.brain.b[key])):
                if random() < self.cross_over_rate:
                    child.brain.b[key][i] = dominant_gene.brain.b[key][i]
                else:
                    child.brain.b[key][i] = recessive_gene.brain.b[key][i]

        # Mutation
        for key in child.brain.w:
            for i in range(0, len(child.brain.w[key])):
                for j in range(0, len(child.brain.w[key][i])):
                    if random() < self.mutation_rate:
                        child.brain.w[key][i, j] = (child.brain.B - child.brain.A) * random() + child.brain.A
                    else:
                        child.brain.w[key][i, j] = (child.brain.B - child.brain.A) * random() + child.brain.A
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
        if Engine.distance(creature, creature.prey) < self.capture_radius:
            self.re_spawn(creature.prey)
            creature.fitness += 100

    def assign_prey(self, creature):
        prey = None

        for candidate in self.entities[1]:
            if prey is None or Engine.distance(creature, candidate) < Engine.distance(creature, prey):
                prey = candidate

        creature.target_prey(prey)

    def save_generation(self, file_name):
        file = open(file_name + '_backup', "w")

        lines = []

        for mouse in self.entities[0]:
            lines.append(str(mouse.brain.w).replace('\n', '').replace(' ', '') + '\n')
            lines.append(str(mouse.brain.b).replace('\n', '').replace(' ', '') + '\n')

        file.writelines(lines)

        file.close()

        os.remove(file_name)
        os.rename(file_name + '_backup', file_name)
