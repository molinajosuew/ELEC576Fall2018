from Engine import Engine
from GUI import GUI
from math import inf

GUI(Engine(world_width = 500,
           world_height = 500,
           n_mice = 10,
           n_cheese = 15,
           n_cats = 0,
           n_generations = inf,
           generation_life = 5000,
           cross_over_rate = .5,
           mutation_rate = .05,
           chromosomes_file = 'chromosomes_file',
           stats_file = 'stats_file'), show = False)
