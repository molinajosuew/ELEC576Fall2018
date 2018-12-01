import pygame


class GUI(object):
    def __init__(self, engine):
        pygame.init()
        screen = pygame.display.set_mode((engine.world_width, engine.world_height))
        generation = 1

        while generation < engine.n_generations:
            print('generation ' + str(generation + 1) + ' of ' + str(engine.n_generations) + ' in progress')

            while engine.tick():
                pass
                # if generation % 10 == 0 or generation == 1:
                #     screen.fill((255, 255, 255))
                #
                #     for entity in engine.entities:
                #         if entity.kind == 'mouse':
                #             pygame.draw.line(screen, (200, 200, 255), (entity.x, entity.y), (entity.prey.x, entity.prey.y))
                #             pygame.draw.line(screen, (255, 200, 200), (entity.x, entity.y), (entity.predator.x, entity.predator.y))
                #         elif entity.kind == 'cat':
                #             pygame.draw.line(screen, (0, 0, 0), (entity.x, entity.y), (entity.prey.x, entity.prey.y))
                #
                #     best_mouse = engine.get_fittest('mouse')
                #     best_cat = engine.get_fittest('cat')
                #
                #     for entity in engine.entities:
                #         if entity.kind == 'mouse':
                #             if entity == best_mouse:
                #                 entity_sprite = pygame.image.load('./images/best_mouse.png')
                #             else:
                #                 entity_sprite = pygame.image.load('./images/mouse.png')
                #         elif entity.kind == 'cat':
                #             if entity == best_cat:
                #                 entity_sprite = pygame.image.load('./images/best_cat.png')
                #             else:
                #                 entity_sprite = pygame.image.load('./images/cat.png')
                #         else:
                #             entity_sprite = pygame.image.load('./images/cheese.png')
                #
                #         sprite_rectangle = entity_sprite.get_rect()
                #         sprite_rectangle.centerx = entity.x
                #         sprite_rectangle.centery = entity.y
                #         screen.blit(entity_sprite, sprite_rectangle)
                #
                #     pygame.display.flip()

            generation += 1
