from NeuralNet import NeuralNet
from env_goal import BioSim
from agent import Agent

env = BioSim()
agent_map = env.agents_map
agent = Agent(agent_map)

NN = NeuralNet()

brain = NeuralNet.create_wiring_from_genome(agent.genome)

actionLevels = agent.brain.feed_forward((agent.position), 128)

def save_frame_state(self) :
    frame_state = []        
    frame_state.append({
        "frame" : self.timestep,
    })
    """"
    for agent in self.agents : 
        frame_state.append({
            "id" : agent.id,
            "alive" : agent.alive,
            "position" : agent.position,
            "color" : agent.color,

            "genome" : [{"sourceType" : g.sourceType, 
                            "sourceNum": g.sourceNum,
                            "targetType" : g.targetType,
                            "targetNum": g.targetNum, 
                            "weight": g.weight
                            } 
                            for g in agent.genome]
            
        })
        """
    frame_state.append({
            "id" : agent.id,
            "position" : agent.position,
            "color" : agent.color,
            "genome" : [{"sourceType" : g.sourceType, 
                            "sourceNum": g.sourceNum,
                            "targetType" : g.targetType,
                            "targetNum": g.targetNum, 
                            "weight": g.weight
                            } 
                            for g in agent.genome]})

    return frame_state
frame = save_frame_state(env)
proba = agent.Prob2Bool(actionLevels[4])
movex = 0.0
movey = 0.0
movex += actionLevels[2]
movex -= actionLevels[3]
movey += actionLevels[0]
movey -= actionLevels[1]

if proba and abs(movex) <= abs(actionLevels[4]) and abs(movey) <= abs(actionLevels[4]):
    print("stay")



print(actionLevels)
print(proba)
