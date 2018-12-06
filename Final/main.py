from Engine import Engine
from GUI import GUI
from math import inf

GUI(Engine(world_width = 1000,
           world_height = 1000,
           n_mice = 20,
           n_cheese = 30,
           n_cats = 10,
           n_generations = inf,
           cross_over_rate = .5,
           mutation_rate = .05,
           file_name = 'mice_generation.txt'))
