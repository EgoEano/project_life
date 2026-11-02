from dataclasses import dataclass, field


@dataclass
class CellCharcterConfig:
    learning_rate: float
    decay_rate: float

@dataclass
class CellConfig:
    state: dict[str, float]
    sensors: list[str]
    actions: list[str]
    character: CellCharcterConfig
    dna_synapses: dict[str, dict[str, float]] = field(default_factory=dict)


class Cell:
    def __init__(self, config: CellConfig) -> None:
        self.state = config.state
        self.character = config.character
        self.sensors = config.sensors
        self.actions = config.actions
        
        self.neuron_activity: dict[str, float] = {}
        self.sensor_activity: dict[str, float] = {}
        self.motor_activity: dict[str, float] = {}
        self.synapses = {src: dict(dsts) for src, dsts in config.dna_synapses.items()}


    def focus(self):
        self.neuron_activity = {}
        self.sensor_activity = {}
        self.motor_activity = {}


    def feel(self, sensory_input: dict[str, float]):
        for k in self.sensors:
            if k in sensory_input:
                self.sensor_activity[k] = sensory_input[k]

        for k, v in self.state.items():
            self.sensor_activity[k] = max(0.0, min(1.0, 1.0 - v))


    def react(self):
        for src, current_signal in self.sensor_activity.items():
            connections = self.synapses.get(src, {})

            for dst, channel_width in connections.items():
                if channel_width > 0:
                    self.motor_activity[dst] = self.motor_activity.get(dst, 0.0) + (current_signal * channel_width)

        for dst in self.motor_activity:
            self.motor_activity[dst] = max(0.0, min(1.0, self.motor_activity[dst]))


    def learn(self):
        all_active_neurons = list(self.neuron_activity.items())

        for i in range(len(all_active_neurons)):
            src, act_src = all_active_neurons[i]
            for j in range(len(all_active_neurons)):
                dst, act_dst = all_active_neurons[j]

                if src == dst: continue
                if src in self.actions: continue

                hebb_change = self.character.learning_rate * (act_src * act_dst)
                current_weight = self.synapses.get(src, {}).get(dst, 0.0)
                hebb_decay = self.character.decay_rate * current_weight

                new_weight = current_weight + (hebb_change - hebb_decay)

                if new_weight > 0:
                    if src not in self.synapses:
                        self.synapses[src] = {}
                    self.synapses[src][dst] = max(0.0, min(1.0, new_weight))
                elif src in self.synapses and dst in self.synapses[src]:
                    del self.synapses[src][dst]


    def live(self, sensory_input: dict[str, float]):
        self.focus()

        self.feel(sensory_input)

        self.react()

        # accumulate all activities
        self.neuron_activity = {**self.sensor_activity, **self.motor_activity}

        # learning stage
        self.learn()

        # selecting top action
        chosen_action = max(self.actions, key=lambda nm: self.neuron_activity.get(nm, 0.0))
        action_power = self.neuron_activity.get(chosen_action, 0.0)

        # selected action
        return chosen_action, action_power


def run():
    cell_state = {
        "energy": 0.5,
    }
    cell_outer_sensors_map = [
        "color"
    ]
    cell_actions = [
        "bite"
    ]
    cell_dna_synapses = {
        "energy": { "color": 0.6, "bite": 0.6 }
    }
    cell_character = CellCharcterConfig(
        learning_rate=0.3,
        decay_rate = 0.02
    )


    cell = Cell(CellConfig(
        state=cell_state,
        sensors=cell_outer_sensors_map, 
        actions=cell_actions,
        character=cell_character,
        dna_synapses=cell_dna_synapses
    ))

    print("energy - 0.5")
    for i in range(5):
        act, power = cell.live({ "color": 0.5 })
        print(act, power)

    cell.state["energy"] = 0.1
    print("energy - 0.1")

    for i in range(5):
        act, power = cell.live({ "color": 0.5 })
        print(act, power)

    print("energy - 1")
    cell.state["energy"] = 1

    for i in range(5):
        act, power = cell.live({ "color": 0.5 })
        print(act, power)
