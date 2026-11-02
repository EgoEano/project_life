from dataclasses import dataclass, field


@dataclass
class CellCharcterConfig:
    learning_rate: float
    decay_rate: float

@dataclass
class CellConfig:
    state: dict[str, float]
    sensors_inner: dict[str, int]
    sensors_outer: dict[str, int]
    actions: dict[str, int]
    character: CellCharcterConfig
    dna_synapses: dict[int, dict[int, float]] = field(default_factory=dict)


class Cell:
    def __init__(self, config: CellConfig) -> None:
        self.state = config.state
        self.character = config.character
        self.sensors_inner_map = config.sensors_inner
        self.sensors_outer_map = config.sensors_outer
        self.action_map = config.actions

        self.neurons_actions: list[int] = list(config.actions.values())
        self.neurons_count = len(self.sensors_inner_map) + len(self.sensors_outer_map) + len(self.action_map)

        # neurons and synapses with DNA preset
        self.neuron_activity = {}
        self.sensor_activity = {}
        self.motor_activity = {}
        self.synapses = {src: dict(dsts) for src, dsts in config.dna_synapses.items()}


    def focus(self):
        # preparing for new activity iteration (may be no need to cut prev feelings?)
        self.neuron_activity = {}
        self.sensor_activity = {}
        self.motor_activity = {}


    def feel(self, sensory_input: dict[str, float]):
        # sensory input into neurons (outer)
        for k, v in self.sensors_outer_map.items():
            self.sensor_activity[v] = sensory_input[k]

        # sensory input into neurons (inner)
        for k, v in self.sensors_inner_map.items():
            self.sensor_activity[v] = max(0.0, min(1.0, 1.0 - self.state[k]))


    def react(self):
        # synapses reaction
        for src, current_signal in self.sensor_activity.items():
            connections = self.synapses.get(src, {})

            for dst, channel_width in connections.items():
                if channel_width > 0:
                    self.motor_activity[dst] = self.motor_activity.get(dst, 0.0) + (current_signal * channel_width)

        # cut values in [0...1] range
        for dst in self.motor_activity:
            self.motor_activity[dst] = max(0.0, min(1.0, self.motor_activity[dst]))


    def learn(self):
        all_active_neurons = list(self.neuron_activity.items())

        for i in range(len(all_active_neurons)):
            src, act_src = all_active_neurons[i]
            for j in range(len(all_active_neurons)):
                dst, act_dst = all_active_neurons[j]

                if src == dst: continue
                # action is not a source for sensor (or it is physical feedback?)
                if src in self.neurons_actions: continue

                hebb_change = self.character.learning_rate * (act_src * act_dst)
                current_weight = self.synapses.get(src, {}).get(dst, 0.0)
                hebb_decay = self.character.decay_rate * current_weight

                new_weight = current_weight + (hebb_change - hebb_decay)

                if new_weight > 0:
                    if src not in self.synapses:
                        self.synapses[src] = {}
                    self.synapses[src][dst] = max(0.0, min(1.0, new_weight))
                elif src in self.synapses and dst in self.synapses[src]:
                    # synaps is dead
                    del self.synapses[src][dst]


    def live(self, sensory_input: dict[str, float]):
        # neuron current state
        self.focus()

        self.feel(sensory_input)

        self.react()

        # accumulate all activities
        self.neuron_activity = {**self.sensor_activity, **self.motor_activity}

        # learning stage
        self.learn()

        # selecting top action
        chosen_action = max(self.neurons_actions, key=lambda idx: self.neuron_activity.get(idx, 0.0))
        action_power = self.neuron_activity.get(chosen_action, 0.0)

        # selected action
        return chosen_action, action_power


def run():
    cell_state = {
        "energy": 0.5,
    }
    cell_inner_sensors_map = {
        "energy": 0,
    }
    cell_outer_sensors_map = {
        "color": 1,
    }
    cell_actions_map = {
        "bite": 2
    }
    cell_dna_synapses = {
        0: { 1: 0.3, 2: 0.3 }
    }
    cell_character = CellCharcterConfig(
        learning_rate=0.3,
        decay_rate = 0.02
    )


    cell = Cell(CellConfig(
        state=cell_state,
        sensors_inner=cell_inner_sensors_map,
        sensors_outer=cell_outer_sensors_map, 
        actions=cell_actions_map,
        character=cell_character,
        dna_synapses=cell_dna_synapses
    ))

    for i in range(10):
        act, power = cell.live({ "color": 0.5 })
        action_name = next(
            name for name, idx in cell_actions_map.items() if idx == act
        )
        print(action_name, power)