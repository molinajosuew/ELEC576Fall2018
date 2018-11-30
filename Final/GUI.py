import pygame


class GUI(object):
    def __init__(self, engine):
        pygame.init()
        screen = pygame.display.set_mode((engine.world_width, engine.world_height))
        generation = 1

        while generation < engine.n_generations:
            print('generation ' + str(generation + 1) + ' of ' + str(engine.n_generations) + ' in progress')

            while engine.tick():
                if generation % 10 == 0 or generation == 1:
                    screen.fill((255, 255, 255))
                    entities = [sub_entity for sub_entities in engine.entities for sub_entity in sub_entities]

                    for entity in entities:
                        if entity.kind == 'mouse':
                            pygame.draw.line(screen, (230, 230, 230), (entity.x, entity.y), (entity.prey.x, entity.prey.y))

                    best_mouse = engine.get_best_mouse()

                    for entity in entities:
                        if entity.kind == 'mouse':
                            if entity == best_mouse:
                                entity_sprite = pygame.image.load('./images/best_mouse.png')
                            else:
                                entity_sprite = pygame.image.load('./images/mouse.png')
                        else:
                            entity_sprite = pygame.image.load('./images/cheese.png')

                        sprite_rectangle = entity_sprite.get_rect()
                        sprite_rectangle.centerx = entity.x
                        sprite_rectangle.centery = entity.y
                        screen.blit(entity_sprite, sprite_rectangle)

                    pygame.display.flip()

            generation += 1
