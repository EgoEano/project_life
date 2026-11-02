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


class NeuronLayer:
    def __init__(
        self, 
        decay_rate: float = 0.05,
        potentials: list[float] = None,
        # thresholds: list[float] = None,
        # timestamps: list[int] = None
    ) -> None:
        self.decay_rate = decay_rate
        
        self.neuron_potentials = potentials if potentials is not None else []
        # self.neuron_thresholds = thresholds if thresholds is not None else []
        # self.neuron_timestamps = timestamps if timestamps is not None else []
  


    def add_neuron(self, threshold: float = 1.0) -> int:
        # после реализации нейрогенеза, сначала ищем есть ли свободные id
        # если есть - используем, а иначе создаем как обычно
        self.neuron_potentials.append(0.0)
        self.neuron_thresholds.append(threshold)
        
        new_neuron_id = len(self.neuron_potentials) - 1
        return new_neuron_id

    def remove_neuron(self, neuron_id: int):
        # удалять элементы из массива нежелательно
        # можно просто занулить нейрон и добавить его id в list нейрогенеза для переиспользования
        pass

    def forward(
        self, 
        input: list[tuple[int, float]], # id: [-1...1]
    ) -> list[tuple[int, float]]: # id: [0...1]

        output: list[tuple[int, float]] = []

        for idx, weight in input:
            if idx >= len(self.neuron_potentials): continue

            self.neuron_potentials[idx] += weight

            if self.neuron_potentials[idx] >= self.neuron_thresholds[idx]:
                output.append((idx, 1.0))
                self.neuron_potentials[idx] = 0.0

        return output


class SynapseLayer:
    def __init__(
        self, 
        sources: list[int] = None,
        targets: list[int] = None,
        weights: list[float] = None
    ) -> None:
        self.synapse_sources = sources if sources is not None else []
        self.synapse_targets = targets if targets is not None else []
        self.synapse_weights = weights if weights is not None else [] # [-1...1]   


    def add_synapse(self, source_idx: int, target_idx: int, weight: float) -> int:
        # обдумать механизм аналогичный нейрогенезу
        self.synapse_sources.append(source_idx)
        self.synapse_targets.append(target_idx)
        self.synapse_weights.append(weight)
        
        new_synapse_id = len(self.synapse_weights) - 1
        return new_synapse_id


    def remove_synapse(self, synapse_id: int):
        # удалять элементы из массива нежелательно
        # можно просто занулить синапс и добавить его id в list нейрогенеза для переиспользования
        pass

    def forward(
        self, 
        input: list[tuple[int, float]]
    ) -> list[tuple[int, float]]:
        outputs: list[tuple[int, float]] = []

        active_dict = {neuron_id: spike_val for neuron_id, spike_val in input if spike_val > 0}
        
        if not active_dict: return outputs

        for i in range(len(self.synapse_weights)):
            weight = self.synapse_weights[i]
            
            # Если синапс "мертв" (вес 0.0), просто пропускаем его
            # А если синапс не мертв а просто между +/-? Он не мертв тогда
            # Определить как обучать такой синапс (не пропускать)
            if weight == 0.0: continue
                
            src = self.synapse_sources[i]

            if src in active_dict:
                dst = self.synapse_targets[i]
                weight_spike = active_dict[src] * weight
                outputs.append((dst, weight_spike))

        return outputs


class SpikingNeuralNetwork:
    def __init__(
        self, 
        neuron_layer: NeuronLayer, 
        synapse_layer: SynapseLayer
    ) -> None:
        self.neurons = neuron_layer
        self.synapses = synapse_layer
        
        # Глобальный счетчик времени симуляции для ленивого leak-эффекта
        self.current_tick: int = 0

    def live_tick(
        self, 
        sensory_input: list[tuple[int, float]]  # [(ID_сенсорного_нейрона, заряд), ...]
    ) -> list[tuple[int, float]]:               # [(ID_моторного_нейрона, 1.0), ...]
        """
        Один полный шаг (такт) жизни нейросети.
        Реализует проброс сигнала: Сенсоры -> Синапсы -> Нейроны -> Выход
        """
        # Шаг 1: Инкрементируем глобальное время
        self.current_tick += 1

        # Шаг 2: Направляем первичные спайки от сенсоров напрямую в слой нейронов.
        # Нейроны накопят этот потенциал (с учетом ленивого затухания) и могут выдать спайки.
        sensor_output_spikes = self.neurons.forward(sensory_input, self.current_tick)

        # Шаг 3: Передаем сгенерированные нейронами спайки в слой синапсов.
        # Синапсы перемножают спайки на веса и находят ID целевых нейронов.
        synaptic_pulses = self.synapses.forward(sensor_output_spikes)

        # Шаг 4: Вторичный проход по нейронам. 
        # Направляем импульсы из синапсов в нейроны-цели, чтобы обновить их потенциалы
        # и собрать финальные спайки текущего такта (включая моторные).
        final_spikes = self.neurons.forward(synaptic_pulses, self.current_tick)

        # Шаг 5: Фильтрация моторных выходов.
        # Возвращаем наружу только те спайки, которые принадлежат моторным нейронам.
        # (Внешняя система клетки сама решит, какие ID привязаны к физическим действиям).
        return final_spikes

    def live_tick_example(self, sensory_input: list[tuple[int, float]]) -> list[tuple[int, float]]:
        self.current_tick += 1

        # 1. Синапсы берут спайки, которые родились НА ПРОШЛОМ ТАКТЕ, 
        #    и превращают их в импульсы для текущего такта.
        synaptic_pulses = self.synapses.forward(self.previous_tick_spikes)

        # 2. Объединяем импульсы от синапсов и свежий ввод от сенсоров
        combined_input = synaptic_pulses + sensory_input

        # 3. Нейроны обрабатывают ВСЕ входящие сигналы одновременно
        current_spikes = self.neurons.forward(combined_input, self.current_tick)

        # Сохраняем спайки для следующего такта (синапсы обработают их на тике Т+1)
        self.previous_tick_spikes = current_spikes

        return current_spikes



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
