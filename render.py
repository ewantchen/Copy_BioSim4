import pygame
import os
import json
from src.params import PARAMS

window_size = PARAMS["WINDOW_SIZE"]
size = PARAMS["SIZE"]
max_time = PARAMS["MAX_TIME"]

def render_generation(gen_number):
    folder = os.path.join(os.path.dirname(__file__), "src","generations")
    with open(os.path.join(folder, f"gen_{gen_number}.json"), "r") as f:
        generation_state = json.load(f)

    if not pygame.get_init() : 
        pygame.init()

    clock = pygame.time.Clock()
    window = pygame.display.set_mode((window_size, window_size))
    canvas = pygame.Surface((window_size, window_size))
    canvas.fill((255, 255, 255))
    pix_square_size = window_size / size

    running = True
    frame_index = 0

    while running and frame_index < max_time:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        window.fill((255, 255, 255))  # Nettoyer la fenêtre à chaque frame

        frame_state = generation_state[frame_index]
        #agents = frame_state[1:]  # Ignorer le premier dict "frame"
        #agents = frame_state

        for agent in frame_state["agents"]:
            x, y = frame_state["agents"][agent]["position"]
            color = frame_state["agents"][agent]["color"]
            pygame.draw.circle(
                window,
                color,
                ((x + 0.5) * pix_square_size, (y + 0.5) * pix_square_size),
                pix_square_size / 2,
            )


        pygame.display.flip()  # Met à jour tout l’écran

        clock.tick(PARAMS["FPS"])  # Contrôle vitesse (fps)

        #print(frame_index)
        frame_index += 1

    pygame.quit()


render_generation(0)
render_generation(1)
render_generation(2)
render_generation(3)
render_generation(4)
render_generation(5)
render_generation(6)
render_generation(7)