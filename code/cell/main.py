

from cell.cell_gen4_5 import Cell, CellCharcterConfig, CellConfig


def makeCell():
    cell_state = {"energy": 1.0, "health": 1.0}

    cell_homeostasis_weights = {
        "energy": {"weight": 1.0, "threshold": 0.005}, 
        "health": {"weight": 1.5, "threshold": 0.001},  
        "toxins": {"weight": -1.2, "threshold": 0.0}
    }

    cell_sensors = [
        "light_greenX+", "light_greenX-", "light_greenY+", "light_greenY-", "light_green_inside",
        "light_redX+",   "light_redX-",   "light_redY+",   "light_redY-",   "light_red_inside",
        "light_blueX+",  "light_blueX-",  "light_blueY+",  "light_blueY-",  "light_blue_inside"
    ]

    old_cell_actions = ["eat", "bite", "moveX+", "moveX-", "moveY+", "moveY-"]

    cell_actions = {
        "rest": 0.0,
        "eat": 0.05, 
        "bite": 0.1, 
        "moveX+": 0.05, 
        "moveX-": 0.05, 
        "moveY+": 0.05, 
        "moveY-": 0.05,
    }

    
    cell_dna_synapses = [
        {"path": ["light_greenX+", 0, "moveX+"], "weights": [0.5, 0.7]},
        {"path": ["light_greenX-", 1, "moveX-"], "weights": [0.5, 0.7]},
        {"path": ["light_greenY+", 2, "moveY+"], "weights": [0.5, 0.7]},
        {"path": ["light_greenY-", 3, "moveY-"], "weights": [0.5, 0.7]},
        {"path": ["light_green_inside", 4, "eat"], "weights": [0.8, 0.8]},
        {"path": ["light_redX+",   5, "moveX-"], "weights": [0.5, 0.7]},
        {"path": ["light_redX-",   6, "moveX+"], "weights": [0.5, 0.7]},
        {"path": ["light_redY+",   7, "moveY-"], "weights": [0.5, 0.7]},
        {"path": ["light_redY-",   8, "moveY+"], "weights": [0.5, 0.7]},
        {"path": ["light_red_inside",   9, "bite"], "weights": [0.8, 0.8]}
    ]

    cell_character = CellCharcterConfig(
        learning_rate=0.07, decay_rate=0.05, tag_decay_rate=0.05,
        base_metabolism=0.002, base_neuron_threshold=0.8,
        spike_cost=0.0005, action_cost=0.01
    )

    return Cell(CellConfig(
        state=cell_state, 
        homeostasis_weights=cell_homeostasis_weights,
        sensors=cell_sensors, 
        actions=cell_actions,
        character=cell_character, 
        dna_synapses=cell_dna_synapses
    ))