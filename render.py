import pygame
import os
import json
from src.params import PARAMS
import matplotlib.pyplot as plt
import matplotlib.animation as animation

window_size = PARAMS["WINDOW_SIZE"]
size = PARAMS["SIZE"]
max_time = PARAMS["MAX_TIME"]
fps = PARAMS["FPS"]
pix_square_size = window_size / size


def load_generation_data(gen_number):
    folder = os.path.join(os.path.dirname(__file__),"src", "generations")
    with open(os.path.join(folder, f"gen_{gen_number}.json"), "r") as f:
        return json.load(f)

#faite uniquement pour montrer en temps réel la simulation
def render_generation(gen_number):
    generation_state = load_generation_data(gen_number)

    if not pygame.get_init() : 
        pygame.init()

    clock = pygame.time.Clock()
    window = pygame.display.set_mode((window_size, window_size))
    canvas = pygame.Surface((window_size, window_size))
    canvas.fill((255, 255, 255))
    

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

#utilisé dans create_video
def render(gen_number):
    generation_state = load_generation_data(gen_number)
    frame_index = 0

    while frame_index < max_time:
        # À chaque frame on crée un canvas vide qu'on retourne à chaque fois.
        canvas = pygame.Surface((window_size, window_size))
        canvas.fill((255,255,255))

        # on récupère toute les données
        frame_state = generation_state["frames"][frame_index]
        pygame.draw.line(canvas,(255,0,0), (14*pix_square_size,71*pix_square_size), (14*pix_square_size,0))
        for agent in frame_state["agents"] : 
            x, y = frame_state["agents"][agent]["position"]
            color = frame_state["agents"][agent]["color"]
            pygame.draw.circle(
                canvas,
                color,
                ((x + 0.5) * pix_square_size, (y + 0.5) * pix_square_size),
                pix_square_size / 2,
            )
        yield canvas

        frame_index += 1


# On crée un fichier mp4 pour montrer la gen
def create_video(frames, output_file, fps = fps ) :
    # On initialise les données pour la fenêtre d'animation
    fig = plt.figure(figsize=(frames[0].get_width()/100, frames[0].get_height()/100))
    ax = fig.add_axes([0,0,1,1])
    ax.axis('off')

    # On initialise la première frame
    img = ax.imshow(pygame.surfarray.pixels3d(frames[0]).swapaxes(0, 1))

    # On défini une fonction qui update la frame
    def update(frame):
        img.set_data(pygame.surfarray.pixels3d(frame).swapaxes(0, 1))
        return [img]

    # On l'anime et ensuite on en fait un fichier
    ani = animation.FuncAnimation(fig, update, frames=frames, blit=True)
    ani.save(output_file, writer='ffmpeg', fps=fps)
    plt.close() 



[0,1,2,50,100,500,1200]
frames = list(render(0))
create_video(frames, "gen_0.mp4")
frames = list(render(1))
create_video(frames, "gen_1.mp4")
frames = list(render(2))
create_video(frames, "gen_2.mp4")
frames = list(render(50))
create_video(frames, "gen_50.mp4")
frames = list(render(100))
create_video(frames, "gen_100.mp4")
frames = list(render(500))
create_video(frames, "gen_500.mp4")
frames = list(render(1200))
create_video(frames, "gen_1200.mp4")







    






