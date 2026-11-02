from collections import defaultdict
from dataclasses import dataclass, field
import math


@dataclass
class CellCharcterConfig:
    learning_rate: float
    decay_rate: float

    base_metabolism: float
    spike_cost: float
    action_cost: float


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
        
        self.free_ids: set[int] = set()

  
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
            self.free_ids.add(neuron_id)


    def forward(
        self, 
        input: list[tuple[int, float]], # id: [-1...1],
        current_time: int
    ) -> list[tuple[int, float]]: # id: [0...1]

        output: list[tuple[int, float]] = []

        # 1. leak memory (decay) for every alive neurons
        # Утечку нужно сделать не проходом по всем, а хранить некое время последнего обращения
        # После запроса считать разницу во времени из neuron_timestamps и высчитывать накопившийся decay
        for idx in range(len(self.neuron_potentials)):
            # Пропускаем мертвые нейроны, не тратим такты CPU
            if idx in self.free_ids: continue

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
                self.neuron_potentials[idx] -= self.neuron_thresholds[idx] # Refractory
                
                if self.neuron_potentials[idx] >= self.neuron_thresholds[idx]:
                    self.neuron_potentials[idx] = 0.0

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
        
        self.free_ids: set[int] = set()

        self.source_to_synapses: dict[int, list[int]] = defaultdict(list)

        if sources:
            for idx, src in enumerate(sources):
                self.source_to_synapses[src].append(idx)


    def add_synapse(self, source_idx: int, target_idx: int, weight: float) -> int:
        if self.free_ids:
            idx = self.free_ids.pop()
            self.synapse_sources[idx] = source_idx
            self.synapse_targets[idx] = target_idx
            self.synapse_weights[idx] = weight

            # Регистрируем синапс в индексной карте
            self.source_to_synapses[source_idx].append(idx)

            return idx

        self.synapse_sources.append(source_idx)
        self.synapse_targets.append(target_idx)
        self.synapse_weights.append(weight)

        idx = len(self.synapse_weights) - 1
        self.source_to_synapses[source_idx].append(idx)

        return idx


    def remove_synapse(self, synapse_id: int):
        if 0 <= synapse_id < len(self.synapse_weights):
            src = self.synapse_sources[synapse_id]
            if synapse_id in self.source_to_synapses[src]:
                self.source_to_synapses[src].remove(synapse_id)

            self.synapse_weights[synapse_id] = 0.0
            self.free_ids.add(synapse_id)


    def forward(
        self, 
        input: list[tuple[int, float]]
    ) -> list[tuple[int, float]]:
        active_dict = {neuron_id: spike_val for neuron_id, spike_val in input if spike_val > 0}

        if not active_dict: return []

        # Словарь для ОПТИМИЗАЦИИ 2: Агрегируем заряды по целевым нейронам { ID_цели: суммарный_заряд }
        aggregated_outputs: dict[int, float] = defaultdict(float)

        # EVENT-DRIVEN ПРОХОД: Перебираем только те нейроны, которые Реально Выдали Спайк
        for src_id, spike_val in active_dict.items():
            # Если для этого нейрона нет синапсов, просто идем дальше
            if src_id not in self.source_to_synapses:
                continue
                
            # Достаем только связанные с ним синапсы
            for syn_idx in self.source_to_synapses[src_id]:
                # Замечание 1: Больше не проверяем "weight == 0.0", так как синапс может ожить при STDP. 
                # Проверяем только то, что синапс физически не удален (нет в free_ids).
                if syn_idx in self.free_ids:
                    continue
                    
                dst_id = self.synapse_targets[syn_idx]
                weight = self.synapse_weights[syn_idx]
                
                # Суммируем импульсы на целевом нейроне
                aggregated_outputs[dst_id] += spike_val * weight

        # Конвертируем агрегированный словарь в компактный список кортежей для NeuronLayer
        return list(aggregated_outputs.items())


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

        # Буфер для хранения спайков, родившихся НА ПРОШЛОМ ТАКТЕ (T)
        # Именно они пройдут через синапсы на текущем такте (T+1)
        self.previous_tick_spikes: list[tuple[int, float]] = []

    def live_tick(
        self, 
        sensory_input: list[tuple[int, float]]  # [(ID_сенсорного_нейрона, заряд), ...]
    ) -> list[tuple[int, float]]:               # [(ID_моторного_нейрона, 1.0), ...]
        """
        Один полный шаг (такт) жизни нейросети.
        Реализует проброс сигнала: Сенсоры -> Синапсы -> Нейроны -> Выход
        """
        # Шаг 1: Инкрементируем глобальное время (ровно один раз за такт)
        self.current_tick += 1

        # Шаг 2: Синапсы берут спайки из ПРОШЛОГО такта и превращают их в импульсы для текущего.
        # Это обеспечивает Propagation Delay (синаптическую задержку).
        synaptic_pulses = self.synapses.forward(self.previous_tick_spikes)

        # Шаг 3: РЕШЕНИЕ УЯЗВИМОСТИ ПЕТЕЛЬ (Виртуальные генераторы спайков).
        # Вместо того чтобы передавать sensory_input как "потенциал" внутрь NeuronLayer,
        # мы объединяем внутренние синаптические импульсы сети и свежие СЕНСОРНЫЕ СПАЙКИ
        # в один общий поток входящих токов. 
        # Таким образом, сенсоры работают как идеальные генераторы импульсов, 
        # не ломая внутренние потенциалы и рефрактерность рабочих нейронов.
        combined_input = synaptic_pulses + sensory_input

        # Шаг 4: Единственный проход по слою нейронов за такт.
        # Нейроны применяют leak (затухание), суммируют combined_input и генерируют новые спайки.
        current_spikes = self.neurons.forward(combined_input, self.current_tick)

        # Шаг 5: Сохраняем сгенерированные спайки. 
        # Синапсы обработают их на следующем такте (Т + 1).
        self.previous_tick_spikes = current_spikes

        # Шаг 6: Возвращаем спайки текущего такта наружу.
        # Внешний класс (Cell) сам отфильтрует их по actions_map, чтобы выполнить моторные действия.
        return current_spikes



class Cell:
    def __init__(self, config: CellConfig) -> None:
        self.current_time = 0

        self.state = config.state
        self.character = config.character
        self.sensors = config.sensors
        self.actions = config.actions

        self.neurons_sensors = NeuronLayer(decay_rate=0.0)
        self.synapses_sensor_to_erste = SynapseLayer()
        self.neurons_erste = NeuronLayer(decay_rate=0.05)
        self.synapses_erste_to_action = SynapseLayer()
        self.neurons_actions = NeuronLayer(decay_rate=0.1)

        # Словари для быстрого поиска ID нейронов по их текстовым именам
        self.sensors_map: dict[str, int] = {}
        self.erste_ids: list[int] = []
        self.actions_map: dict[str, int] = {}

        # Наполняем сенсорный слой
        for name in self.sensors:
            self.sensors_map[name] = self.neurons_sensors.add_neuron(threshold=0.8)    
        # Создаем 3 скрытых интернейрона
        for _ in range(3):
            self.erste_ids.append(self.neurons_erste.add_neuron(threshold=0.3))
        # Наполняем моторный слой
        for name in self.actions:
            self.actions_map[name] = self.neurons_actions.add_neuron(threshold=1.0)

        # 3. ЭКСПРЕССИЯ ДНК (Прокладка транзитных дорожек)
        # config.dna_synapses теперь ожидается как список путей
        if hasattr(config, 'dna_synapses') and config.dna_synapses:
            for gene in config.dna_synapses:
                path = gene["path"]
                weights = gene["weights"]
                
                # Извлекаем компоненты пути
                src_name = path[0]  # Ожидаем строку, например "energy_low"
                hidden_idx = path[1] # Ожидаем число, относительный индекс скрытого нейрона
                dst_name = path[2]  # Ожидаем строку, например "bite"
                
                # Валидация: проверяем, существуют ли такие сенсоры и моторы в текущем теле клетки
                if src_name in self.sensors_map and dst_name in self.actions_map:
                    s_id = self.sensors_map[src_name]
                    a_id = self.actions_map[dst_name]
                    
                    # Маппим относительный индекс из ДНК на реальный ID созданного нейрона в скрытом слое
                    if 0 <= hidden_idx < len(self.erste_ids):
                        h_id = self.erste_ids[hidden_idx]
                        
                        # Протягиваем физические провода через транзитные слои синапсов
                        w_input = weights[0]
                        w_output = weights[1]
                        
                        input_exists = False
                        if s_id in self.synapses_sensor_to_erste.source_to_synapses:
                            for syn_idx in self.synapses_sensor_to_erste.source_to_synapses[s_id]:
                                if self.synapses_sensor_to_erste.synapse_targets[syn_idx] == h_id:
                                    input_exists = True
                                    break
                        
                        if not input_exists:
                            self.synapses_sensor_to_erste.add_synapse(source_idx=s_id, target_idx=h_id, weight=w_input)
                        
                        # --- ПРОКЛАДКА ВЫХОДНОГО СИНАПСА (ЗАЩИТА ОТ ДУБЛИКАТОВ) ---
                        # Проверяем, нет ли уже синапса от h_id к a_id (ведь узел 0 используется дважды)
                        output_exists = False
                        if h_id in self.synapses_erste_to_action.source_to_synapses:
                            for syn_idx in self.synapses_erste_to_action.source_to_synapses[h_id]:
                                if self.synapses_erste_to_action.synapse_targets[syn_idx] == a_id:
                                    output_exists = True
                                    break
                        
                        if not output_exists:
                            self.synapses_erste_to_action.add_synapse(source_idx=h_id, target_idx=a_id, weight=w_output)



    def feel(self, sensory_input: dict[str, float]) -> list[tuple[int, float]]:
        output = []

        # 1. Обработка внешних сенсоров
        for name, value in sensory_input.items():
            if name in self.sensors_map:
                s_id = self.sensors_map[name]
                # Простая кодировка: если значение выше 0.5, сенсорный нейрон гарантированно выдает спайк
                if value >= 0.5:
                    output.append((s_id, value))

        # 2. Обработка внутренних потребностей (метаболизм)
        for name, value in self.state.items():
            if name in self.sensors_map:
                s_id = self.sensors_map[name]
                # Например, инвертируем: чем меньше значение ресурса, тем сильнее "голод"
                inverted_value = max(0.0, min(1.0, 1.0 - value))
                if inverted_value >= 0.5:
                    output.append((s_id, inverted_value))

        return output


    def learn_layer(
        self, 
        synapse_layer: SynapseLayer, 
        src_neurons: NeuronLayer, 
        dst_neurons: NeuronLayer, 
        active_src_ids: set[int], 
        active_dst_ids: set[int]
    ):
        """
        Универсальный STDP: обучает переданный синаптический слой 
        на основе таймстампов связанных слоев нейронов.
        """
        tau = 3.0 
        
        for i in range(len(synapse_layer.synapse_weights)):
            # ОПТИМИЗАЦИЯ: Если синапс удален, вообще его не трогаем
            if i in synapse_layer.free_ids:
                continue
                
            src_id = synapse_layer.synapse_sources[i]
            dst_id = synapse_layer.synapse_targets[i]
            
            # Проверяем активность на текущем такте
            if src_id not in active_src_ids and dst_id not in active_dst_ids:
                continue
                
            t_src = src_neurons.neuron_timestamps[src_id]
            t_dst = dst_neurons.neuron_timestamps[dst_id]
            
            # Если кто-то из пары нейронов еще ни разу не стрелял за историю вселенной
            if t_src < 0 or t_dst < 0:
                continue
                
            dt = t_dst - t_src
            
            # Защита от обучения на слишком старых, архивных спайках
            if abs(dt) > tau * 2: 
                continue

            # Экспоненциальное правило Hebbian STDP
            if dt > 0:
                dw = self.character.learning_rate * math.exp(-dt / tau)
            elif dt < 0:
                dw = -self.character.learning_rate * math.exp(dt / tau)
            else:
                dw = self.character.learning_rate * 0.5
                
            # Обновление и биологические рамки [-1.0 ... 1.0]
            new_weight = synapse_layer.synapse_weights[i] + dw
            synapse_layer.synapse_weights[i] = max(-1.0, min(1.0, new_weight))


    def choose(self, active_dsts: set[int]) -> tuple[str | None, float]:
        chosen_action = None
        max_potential = -1.0
        
        active_action_names = [name for name, a_id in self.actions_map.items() if a_id in active_dsts]
        
        if active_action_names:
            chosen_action = active_action_names[0]
            max_potential = 1.0
        else:
            # Если спайков нет, выбираем по наивысшему накопленному потенциалу
            for name, a_id in self.actions_map.items():
                pot = self.neurons_actions.neuron_potentials[a_id]
                if pot > max_potential:
                    max_potential = pot
                    chosen_action = name

        # Списания энергии за физическое действие
        if chosen_action is not None:
            self.state["energy"] -= self.character.action_cost

        return chosen_action, max_potential


    def live(self, sensory_input: dict[str, float]) -> tuple[str, float]:
        self.current_time += 1
        
        # 0. Списание базового метаболизма (простой тоже стоит АТФ)
        self.state["energy"] -= self.character.base_metabolism

        # 1.1. Считываем датчики и генерируем спайки на сенсорном слое
        sensor_inputs = self.feel(sensory_input)
        sensor_spikes = self.neurons_sensors.forward(sensor_inputs, self.current_time)
        
        # 1.2. Проводим спайки через синапсы к скрытому слою
        erste_inputs = self.synapses_sensor_to_erste.forward(sensor_spikes)
        erste_spikes = self.neurons_erste.forward(erste_inputs, self.current_time)
        
        # 1.3. Проводим спайки через синапсы к моторному слою
        motor_inputs = self.synapses_erste_to_action.forward(erste_spikes)
        motor_spikes = self.neurons_actions.forward(motor_inputs, self.current_time)
        
        # --- ИНТЕГРАЦИЯ АТФ ЗА МЫШЛЕНИЕ ---
        # Считаем суммарное количество спайков во всей сети за этот такт
        total_spikes = len(sensor_spikes) + len(erste_spikes) + len(motor_spikes) 
        self.state["energy"] -= total_spikes * self.character.spike_cost

        # 3.1. Выделяем множества Свежих спайков для изоляции STDP
        active_sensors = {idx for idx, val in sensor_spikes if val > 0}
        active_erste = {idx for idx, val in erste_spikes if val > 0}
        active_actions = {idx for idx, val in motor_spikes if val > 0}


        # 3.2. Запускаем обучение STDP на основе зафиксированных спайков
        self.learn_layer(
            synapse_layer=self.synapses_sensor_to_erste,
            src_neurons=self.neurons_sensors,
            dst_neurons=self.neurons_erste,
            active_src_ids=active_sensors,
            active_dst_ids=active_erste
        )
        self.learn_layer(
            synapse_layer=self.synapses_erste_to_action,
            src_neurons=self.neurons_erste,
            dst_neurons=self.neurons_actions,
            active_src_ids=active_erste,
            active_dst_ids=active_actions
        )

        # 4. Выбор действия на основе того, у какого моторного нейрона сейчас наивысший потенциал
        chosen_action, max_potential = self.choose(active_actions)

        self.state["energy"] = max(0.0, min(1.0, self.state["energy"]))

        return chosen_action, max_potential




def run():
    cell_state = {
        "energy": 1.0,
    }
    cell_sensors = [
        "energy_low", "light"
    ]
    cell_actions = [
        "bite"
    ]
    cell_dna_synapses = [
        # Врожденный инстинкт 1: Голод -> Промежуточный узел 0 -> Укус
        {
            "path": ["energy_low", 0, "bite"], 
            "weights": [0.8, 0.8]  # Вес синапса 1 (входной) и синапса 2 (выходной)
        },
        # Врожденный инстинкт 2: Свет -> Промежуточный узел 1 -> Укус (например, если хотим сделать слабую стартовую связь)
        {
            "path": ["light", 0, "bite"],
            "weights": [0.1, 0.8]
        }
    ]

    cell_character = CellCharcterConfig(
        learning_rate=0.3,
        decay_rate=0.02,
        base_metabolism=0.01,
        spike_cost=0.001,
        action_cost=0.05
    )

    cell = Cell(CellConfig(
        state=cell_state,
        sensors=cell_sensors, 
        actions=cell_actions,
        character=cell_character,
        dna_synapses=cell_dna_synapses
    ))

    # Функция для красивого вывода весов всей двухслойной сети
    def print_all_weights():
        print("--- Слои синапсов: Сенсоры -> Erste ---")
        for i in range(len(cell.synapses_sensor_to_erste.synapse_weights)):
            src = cell.synapses_sensor_to_erste.synapse_sources[i]
            dst = cell.synapses_sensor_to_erste.synapse_targets[i]
            src_name = [k for k, v in cell.sensors_map.items() if v == src][0]
            # dst - это ID скрытого нейрона, выводим его как число
            print(f"  Синапс [{src_name} -> Скрытый_{dst}]: вес = {cell.synapses_sensor_to_erste.synapse_weights[i]:.2f}")
            
        print("--- Слои синапсов: Erste -> Actions ---")
        for i in range(len(cell.synapses_erste_to_action.synapse_weights)):
            src = cell.synapses_erste_to_action.synapse_sources[i]
            dst = cell.synapses_erste_to_action.synapse_targets[i]
            dst_name = [k for k, v in cell.actions_map.items() if v == dst][0]
            # src - это ID скрытого нейрона
            print(f"  Синапс [Скрытый_{src} -> {dst_name}]: вес = {cell.synapses_erste_to_action.synapse_weights[i]:.2f}")

    print("=== СТАРТОВЫЕ ВЕСА СИНАПСОВ ===")
    print_all_weights()

    print("\n=== ЭТАП 1: Клетка просто видит свет (5 тактов) ===")
    for t in range(5):
        action, potential = cell.live({"energy_low": 0.0, "light": 1.0})
        print(f"Такт {cell.current_time} | Энергия АТФ: {cell.state['energy']:.3f} | Ввод: Свет | Выбрано действие: {action} (потенциал: {potential:.2f})")

    print("\n=== ЭТАП 2: Клетка голодна И видит свет (5 тактов) ===")
    for t in range(5):
        action, potential = cell.live({"energy_low": 1.0, "light": 1.0})
        print(f"Такт {cell.current_time} | Энергия АТФ: {cell.state['energy']:.3f} | Ввод: Голод + Свет | Выбрано действие: {action} (потенциал: {potential:.2f})")

    print("\n=== ФИНАЛЬНЫЕ ВЕСА СИНАПСОВ ===")
    print_all_weights()

    print("\n=== ЭТАП 3: Проверка рефлекса. Снова только свет (2 такта) ===")
    for t in range(2):
        action, potential = cell.live({"energy_low": 0.0, "light": 1.0})
        print(f"Такт {cell.current_time} | Энергия АТФ: {cell.state['energy']:.3f} | Ввод: Только Свет | Выбрано действие: {action} (потенциал: {potential:.2f})")
