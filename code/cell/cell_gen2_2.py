from dataclasses import dataclass, field
import math


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
        thresholds: list[float] = None,
        timestamps: list[int] = None
    ) -> None:
        self.decay_rate = decay_rate
        
        self.neuron_potentials = potentials if potentials is not None else []
        self.neuron_thresholds = thresholds if thresholds is not None else []
        self.neuron_timestamps = timestamps if timestamps is not None else []
        self.free_ids: list[int] = [] # Для будущего нейрогенеза

  
    def add_neuron(self, threshold: float = 1.0) -> int:
        if self.free_ids:
            new_neuron_id = self.free_ids.pop()
            self.neuron_potentials[new_neuron_id] = 0.0
            self.neuron_thresholds[new_neuron_id] = threshold
            self.neuron_timestamps[new_neuron_id] = -1000
            return new_neuron_id
        
        self.neuron_potentials.append(0.0)
        self.neuron_thresholds.append(threshold)
        self.neuron_timestamps.append(-1000)
        return len(self.neuron_potentials) - 1


    def remove_neuron(self, neuron_id: int):
        if 0 <= neuron_id < len(self.neuron_potentials):
            self.neuron_potentials[neuron_id] = 0.0
            self.neuron_thresholds[neuron_id] = 0.0
            self.free_ids.append(neuron_id)


    def forward(
        self, 
        input: list[tuple[int, float]], # id: [-1...1],
        current_time: int
    ) -> list[tuple[int, float]]: # id: [0...1]

        output: list[tuple[int, float]] = []

        # 1. leak memory (decay) for every alive neurons
        for idx in range(len(self.neuron_potentials)):
            self.neuron_potentials[idx] *= (1.0 - self.decay_rate)

        # 2. new inputs
        for idx, weight in input:
            if idx >= len(self.neuron_potentials) or idx in self.free_ids: continue

            self.neuron_potentials[idx] += weight

        # 3. checking thresholds and generating spikes
        for idx in range(len(self.neuron_potentials)):
            if idx in self.free_ids: continue

            if self.neuron_potentials[idx] >= self.neuron_thresholds[idx]:
                output.append((idx, 1.0))
                self.neuron_potentials[idx] = 0.0 # Refractory
                self.neuron_timestamps[idx] = current_time

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
        self.free_ids: list[int] = []


    def add_synapse(self, source_idx: int, target_idx: int, weight: float) -> int:
        if self.free_ids:
            idx = self.free_ids.pop()
            self.synapse_sources[idx] = source_idx
            self.synapse_targets[idx] = target_idx
            self.synapse_weights[idx] = weight
            return idx

        self.synapse_sources.append(source_idx)
        self.synapse_targets.append(target_idx)
        self.synapse_weights.append(weight)
        return len(self.synapse_weights) - 1


    def remove_synapse(self, synapse_id: int):
        if 0 <= synapse_id < len(self.synapse_weights):
            self.synapse_weights[synapse_id] = 0.0
            self.free_ids.append(synapse_id)


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
        self.current_time = 0

        self.state = config.state
        self.character = config.character
        self.sensors = config.sensors
        self.actions = config.actions

        self.sensor_neurons = NeuronLayer(decay_rate=0.0)
        self.sensor_synapses = SynapseLayer()
        self.action_neurons = NeuronLayer(decay_rate=0.1)

        # Словари для быстрого поиска ID нейронов по их текстовым именам
        self.sensors_map: dict[str, int] = {}
        self.actions_map: dict[str, int] = {}

        # Наполняем сенсорный слой
        for name in self.sensors:
            self.sensors_map[name] = self.sensor_neurons.add_neuron(threshold=0.8)
            
        # Наполняем моторный слой
        for name in self.actions:
            self.actions_map[name] = self.action_neurons.add_neuron(threshold=1.0)

        for s_name, s_id in self.sensors_map.items():
            for a_name, a_id in self.actions_map.items():
                self.sensor_synapses.add_synapse(source_idx=s_id, target_idx=a_id, weight=0.1)


    def feel(self, sensory_input: dict[str, float]):
        for k in self.sensors:
            if k in sensory_input:
                self.sensor_activity[k] = sensory_input[k]

        for k, v in self.state.items():
            self.sensor_activity[k] = max(0.0, min(1.0, 1.0 - v))


    def feel(self, sensory_input: dict[str, float]) -> list[tuple[int, float]]:
        output = []
        for name, value in sensory_input.items():
            if name in self.sensors_map:
                s_id = self.sensors_map[name]
                # Простая кодировка: если значение выше 0.5, сенсорный нейрон гарантированно выдает спайк
                if value >= 0.5:
                    output.append((s_id, value))

        return output

    #FIXME РАЗОБРАТЬСЯ КАК РАБОТАЕТ!
    def learn(self):
        """Реализация правила STDP по таймстампам спайков"""
        # Окно времени, внутри которого работает пластичность (например, 3 такта)
        tau = 3.0 
        
        for i in range(len(self.sensor_synapses.synapse_weights)):
            src_id = self.sensor_synapses.synapse_sources[i]
            dst_id = self.sensor_synapses.synapse_targets[i]
            
            # Получаем время последних спайков пре- и пост-синаптического нейронов
            t_src = self.sensor_neurons.neuron_timestamps[src_id]
            t_dst = self.action_neurons.neuron_timestamps[dst_id]
            
            # Разница во времени
            dt = t_dst - t_src
            
            # Если оба нейрона никогда толком не стреляли, пропускаем
            if t_src < 0 or t_dst < 0:
                continue
                
            # Вычисляем изменение веса по классической формуле экспоненциального STDP
            if dt > 0:
                # Пре-синапс выстрелил ДО пост-синапса -> Укрепляем связь
                dw = self.character.learning_rate * math.exp(-dt / tau)
            elif dt < 0:
                # Пре-синапс выстрелил ПОСЛЕ пост-синапса -> Ослабляем связь
                dw = -self.character.learning_rate * math.exp(dt / tau)
            else:
                # Выстрелили одновременно на одном такте
                dw = self.character.learning_rate * 0.5
                
            # Обновляем вес синапса и удерживаем его в биологических рамках [-1.0 ... 1.0]
            # (Отрицательный вес превратит синапс в тормозящий!)
            new_weight = self.sensor_synapses.synapse_weights[i] + dw
            self.sensor_synapses.synapse_weights[i] = max(-1.0, min(1.0, new_weight))



    def live(self, sensory_input: dict[str, float]) -> tuple[str, float]:
        self.current_time += 1
        
        # 1. Считываем датчики и генерируем спайки на сенсорном слое
        sensor_inputs = self.feel(sensory_input)
        sensor_spikes = self.sensor_neurons.forward(sensor_inputs, self.current_time)
        
        # 2. Проводим спайки через синапсы к моторному слою
        motor_inputs = self.sensor_synapses.forward(sensor_spikes)
        motor_spikes = self.action_neurons.forward(motor_inputs, self.current_time)
        
        # 3. Запускаем обучение STDP на основе зафиксированных спайков
        self.learn()
        
        # 4. Выбор действия на основе того, у какого моторного нейрона сейчас наивысший потенциал
        chosen_action = None
        max_potential = -1.0
        
        for name, a_id in self.actions_map.items():
            pot = self.action_neurons.neuron_potentials[a_id]
            if pot > max_potential:
                max_potential = pot
                chosen_action = name
                
        # Если в этом такте у какого-то действия вылетел реальный спайк, 
        # мы можем пометить его как "приоритетное"
        active_spikes = [name for name, a_id in self.actions_map.items() if (a_id, 1.0) in motor_spikes]
        if active_spikes:
            chosen_action = active_spikes[0] # Берем первый спайкнувший
            max_potential = 1.0
            
        return chosen_action, max_potential


def run():
    cell_state = {
        "energy": 0.5,
    }
    cell_sensors = [
        "energy_low", "light"
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
        sensors=cell_sensors, 
        actions=cell_actions,
        character=cell_character,
        dna_synapses=cell_dna_synapses
    ))

    # Выведем стартовые веса синапсов для наглядности
    print("=== СТАРТОВЫЕ ВЕСА СИНАПСОВ ===")
    for i in range(len(cell.sensor_synapses.synapse_weights)):
        src = cell.sensor_synapses.synapse_sources[i]
        dst = cell.sensor_synapses.synapse_targets[i]
        # Ищем текстовые имена для красивого вывода
        src_name = [k for k, v in cell.sensors_map.items() if v == src][0]
        dst_name = [k for k, v in cell.actions_map.items() if v == dst][0]
        print(f"Синапс [{src_name} -> {dst_name}]: вес = {cell.sensor_synapses.synapse_weights[i]:.2f}")

    print("\n=== ЭТАП 1: Клетка просто видит свет (5 тактов) ===")
    # Голода нет (0.0), свет горит (1.0). 
    # Свет сам по себе не должен вызывать укус, так как его синапс еще слаб (0.1)
    for t in range(5):
        action, potential = cell.live({"energy_low": 0.0, "light": 1.0})
        print(f"Такт {cell.current_time} | Ввод: Свет | Выбрано действие: {action} (потенциал: {potential:.2f})")

    print("\n=== ЭТАП 2: Клетка голодна И видит свет (5 тактов) ===")
    # Голод высокий (1.0), свет горит (1.0).
    # Голод активирует укус (сигнал сильный), а свет бьет одновременно.
    # STDP должен зафиксировать совпадение спайков!
    for t in range(5):
        action, potential = cell.live({"energy_low": 1.0, "light": 1.0})
        print(f"Такт {cell.current_time} | Ввод: Голод + Свет | Выбрано действие: {action} (потенциал: {potential:.2f})")

    print("\n=== ФИНАЛЬНЫЕ ВЕСА СИНАПСОВ ===")
    # Проверяем, обучилась ли клетка
    for i in range(len(cell.sensor_synapses.synapse_weights)):
        src = cell.sensor_synapses.synapse_sources[i]
        dst = cell.sensor_synapses.synapse_targets[i]
        src_name = [k for k, v in cell.sensors_map.items() if v == src][0]
        dst_name = [k for k, v in cell.actions_map.items() if v == dst][0]
        print(f"Синапс [{src_name} -> {dst_name}]: вес = {cell.sensor_synapses.synapse_weights[i]:.2f}")

    print("\n=== ЭТАП 3: Проверка рефлекса. Снова только свет (2 такта) ===")
    # Проверяем, будет ли клетка кусаться от ОДНОГО ТОЛЬКО СВЕТА теперь, когда связь выросла
    for t in range(2):
        action, potential = cell.live({"energy_low": 0.0, "light": 1.0})
        print(f"Такт {cell.current_time} | Ввод: Только Свет | Выбрано действие: {action} (потенциал: {potential:.2f})")
