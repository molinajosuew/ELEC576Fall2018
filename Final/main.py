from Engine import Engine
from GUI import GUI
from math import inf

GUI(Engine(world_width = 600,
           world_height = 600,
           n_mice = 10,
           n_cheese = 20,
           n_cats = 10,
           n_generations = inf,
           cross_over_rate = .75,
           mutation_rate = .01,
           file_name = 'mice_generation.txt'))
