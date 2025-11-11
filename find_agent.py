import json

scale = 1
x1, y1 = 90,20
x2, y2 = 90,90


# Charger ton fichier JSON
with open("src/generations/gen_50.json") as f:
    generation_state= json.load(f)
frame_index = 291
frame_state = generation_state["frames"][frame_index]

for agent in frame_state["agents"]:
    pos_x, pos_y = frame_state["agents"][agent]["position"]
    if x1 <= pos_x <= x2 and y1 <= pos_y <= y2:
        print(agent)

print("rien")

            