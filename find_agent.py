import json

scale = 128 / 512
x1, y1 =278 * scale ,250 * scale
x2, y2 = 296 * scale,263 *scale


# Charger ton fichier JSON
with open("src/generations/gen_0.json") as f:
    generation_state= json.load(f)
frame_index = 0
frame_state = generation_state[frame_index]

for agent in frame_state["agents"]:
    pos_x, pos_y = frame_state["agents"][agent]["position"]
    if x1 <= pos_x <= x2 and y1 <= pos_y <= y2:
        print(agent)

print("rien")

            