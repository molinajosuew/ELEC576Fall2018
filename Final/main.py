from Engine import Engine
from GUI import GUI
from math import inf

GUI(Engine(world_width = 700,
           world_height = 700,
           n_mice = 10,
           n_cheese = 10,
           n_cats = 10,
           n_generations = inf,
           generation_life = 5000,
           cross_over_rate = .5,
           mutation_rate = .01,
           chromosomes_file = 'cats_c',
           stats_file = 'cats_s'),
    show = True)
