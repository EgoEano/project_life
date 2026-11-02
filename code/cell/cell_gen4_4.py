from collections import defaultdict
from dataclasses import dataclass, field
import math


@dataclass
class CellCharcterConfig:
    learning_rate: float
    decay_rate: float
    tag_decay_rate: float

    base_metabolism: float
    base_neuron_threshold: float
    spike_cost: float
    #FIXME action_cost не должен закладываться тут а считаться из действия 
    action_cost: float


@dataclass
class CellConfig:
    state: dict[str, float]
    homeostasis_weights: dict[str, dict[str, float]]
    sensors: list[str]
    actions: list[str]
    character: CellCharcterConfig
    dna_synapses: dict[str, dict[str, float]] = field(default_factory=list)


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
        
        self.neuron_spike_timestamps = [-100] * len(self.neuron_potentials)
        self.neuron_decays = [decay_rate] * len(self.neuron_potentials)
        self.neuron_tags = [0.0] * len(self.neuron_potentials)

        self.free_ids: set[int] = set()
        self.active_neuron_tags: set[int] = set()

  
    def add_neuron(self, threshold: float = 1.0, decay_rate: float = 0.05) -> int:
        if self.free_ids:
            new_neuron_id = self.free_ids.pop()

            self.neuron_potentials[new_neuron_id] = 0.0
            self.neuron_thresholds[new_neuron_id] = threshold
            self.neuron_timestamps[new_neuron_id] = 0
            self.neuron_spike_timestamps[new_neuron_id] = -100
            self.neuron_decays[new_neuron_id] = decay_rate
            self.neuron_tags[new_neuron_id] = 0.0

            return new_neuron_id
        
        self.neuron_potentials.append(0.0)
        self.neuron_thresholds.append(threshold)
        self.neuron_timestamps.append(0)
        self.neuron_spike_timestamps.append(-100)
        self.neuron_decays.append(decay_rate)
        self.neuron_tags.append(0.0)

        return len(self.neuron_potentials) - 1


    def remove_neuron(self, neuron_id: int):
        if 0 <= neuron_id < len(self.neuron_potentials):
            self.neuron_potentials[neuron_id] = 0.0
            self.neuron_thresholds[neuron_id] = 0.0
            self.neuron_decays[neuron_id] = 0.0
            self.neuron_timestamps[neuron_id] = 0
            self.neuron_spike_timestamps[neuron_id] = -100
            self.neuron_tags[neuron_id] = 0.0
            if neuron_id in self.active_neuron_tags:
                self.active_neuron_tags.remove(neuron_id)

            self.free_ids.add(neuron_id)


    def clear_and_decay_neuron_tags(self):
        """Ежетактовое остывание следов памяти нейронов (вызывается в начале такта)"""
        expired = []
        for idx in self.active_neuron_tags:
            self.neuron_tags[idx] *= 0.9  # FIXME Вынести в конфиг
            if self.neuron_tags[idx] < 0.01: # FIXME Вынести в конфиг
                self.neuron_tags[idx] = 0.0
                expired.append(idx)
        for idx in expired:
            self.active_neuron_tags.remove(idx)


    def all_neurons_or_self_timestamps(self, idx: int, current_time: int) -> int:
        """Вспомогательный метод ленивого leak-обновления перед чтением состояния нейрона"""
        dt = current_time - self.neuron_timestamps[idx]
        if dt > 0:
            self.neuron_potentials[idx] *= math.pow(1.0 - self.neuron_decays[idx], dt)
            self.neuron_timestamps[idx] = current_time
        return current_time


    def forward(
        self, 
        input: list[tuple[int, float]], # id: [-1...1],
        current_time: int
    ) -> list[tuple[int, float]]: # id: [0...1]

        output: list[tuple[int, float]] = []
        updated_neurons = set()

        # 0. Остужаем теги памяти нейронов перед шагом
        self.clear_and_decay_neuron_tags()

        # 1. new inputs
        for idx, weight in input:
            if idx >= len(self.neuron_potentials) or idx in self.free_ids: continue

            self.all_neurons_or_self_timestamps(idx, current_time)
            self.neuron_potentials[idx] += weight
            updated_neurons.add(idx)

        # 2. checking thresholds and generating spikes
        for idx in range(len(self.neuron_potentials)):
            if idx in self.free_ids: continue

            if idx not in updated_neurons:
                self.all_neurons_or_self_timestamps(idx, current_time)

            if self.neuron_potentials[idx] >= self.neuron_thresholds[idx]:
                output.append((idx, 1.0))

                self.neuron_potentials[idx] -= self.neuron_thresholds[idx]
                
                if self.neuron_potentials[idx] >= self.neuron_thresholds[idx]:
                    self.neuron_potentials[idx] = 0.0

                self.neuron_spike_timestamps[idx] = current_time
                self.neuron_tags[idx] = 1.0
                self.active_neuron_tags.add(idx)

        return output


class SynapseLayer:
    def __init__(
        self,
        tag_decay_rate: float,
        sources: list[int] = None,
        targets: list[int] = None,
        weights: list[float] = None
    ) -> None:
        self.tag_decay_rate = tag_decay_rate

        self.synapse_sources = sources if sources is not None else []
        self.synapse_targets = targets if targets is not None else []
        self.synapse_weights = weights if weights is not None else [] # [-1...1]   

        self.synapse_tags: list[float] = [0.0] * len(self.synapse_weights) 
        self.active_tags: set[int] = set()
        self.free_ids: set[int] = set()

        self.source_to_synapses: dict[int, list[int]] = defaultdict(list)
        self.target_to_synapses: dict[int, list[int]] = defaultdict(list)

        if sources:
            for idx, (src, dst) in enumerate(zip(sources, targets)):
                self.source_to_synapses[src].append(idx)
                self.target_to_synapses[dst].append(idx)


    def add_synapse(self, source_idx: int, target_idx: int, weight: float) -> int:
        if self.free_ids:
            idx = self.free_ids.pop()
            self.synapse_sources[idx] = source_idx
            self.synapse_targets[idx] = target_idx
            self.synapse_weights[idx] = weight
            self.synapse_tags[idx] = 0.0

            # Регистрируем синапс в индексной карте
            self.source_to_synapses[source_idx].append(idx)
            self.target_to_synapses[target_idx].append(idx)

            return idx

        self.synapse_sources.append(source_idx)
        self.synapse_targets.append(target_idx)
        self.synapse_weights.append(weight)
        self.synapse_tags.append(0.0)

        idx = len(self.synapse_weights) - 1
        self.source_to_synapses[source_idx].append(idx)
        self.target_to_synapses[target_idx].append(idx)

        return idx


    def remove_synapse(self, synapse_id: int):
        if 0 <= synapse_id < len(self.synapse_weights):
            src = self.synapse_sources[synapse_id]
            dst = self.synapse_targets[synapse_id]

            if synapse_id in self.source_to_synapses[src]:
                self.source_to_synapses[src].remove(synapse_id)
            if synapse_id in self.target_to_synapses[dst]:
                self.target_to_synapses[dst].remove(synapse_id)

            self.synapse_weights[synapse_id] = 0.0
            self.synapse_tags[synapse_id] = 0.0

            if synapse_id in self.active_tags:
                self.active_tags.remove(synapse_id)
            
            self.free_ids.add(synapse_id)


    def forward(
        self, 
        input: list[tuple[int, float]]
    ) -> list[tuple[int, float]]:
        # --- ПОКОЛЕНИЕ 4.0: ЕЖЕТАКТОВОЕ ОСТЫВАНИЕ СЛЕДОВ ПАМЯТИ ---
        expired_tags = []
        for idx in self.active_tags:
            self.synapse_tags[idx] *= (1 - self.tag_decay_rate)
            # Если тег стал пренебрежимо мал, помечаем на удаление из "горячего" списка
            if self.synapse_tags[idx] < 0.001: 
                self.synapse_tags[idx] = 0.0
                expired_tags.append(idx)
            
        for idx in expired_tags:
            self.active_tags.remove(idx)

        # for i in range(len(self.synapse_tags)):
        #     if i in self.free_ids: continue
        #     self.synapse_tags[i] *= 0.8

        active_dict = {neuron_id: spike_val for neuron_id, spike_val in input if spike_val > 0}

        if not active_dict: return []

        aggregated_outputs: dict[int, float] = defaultdict(float)

        # EVENT-DRIVEN: Перебираем только те нейроны, которые Реально Выдали Спайк
        for src_id, spike_val in active_dict.items():
            if src_id not in self.source_to_synapses: continue
                
            # Достаем только связанные с ним синапсы
            for syn_idx in self.source_to_synapses[src_id]:
                if syn_idx in self.free_ids: continue
                    
                # --- ПОКОЛЕНИЕ 4.0: ВЗВОДИМ СИНАПТИЧЕСКИЙ ТЕГ ---
                self.synapse_tags[syn_idx] = 1.0
                self.active_tags.add(syn_idx)

                dst_id = self.synapse_targets[syn_idx]
                weight = self.synapse_weights[syn_idx]
                
                # Суммируем импульсы на целевом нейроне
                aggregated_outputs[dst_id] += spike_val * weight

        return list(aggregated_outputs.items())


class SpikingNeuralNetwork:
    def __init__(
        self, 
        neuron_layer: NeuronLayer, 
        synapse_layer: SynapseLayer,
        neuron_ids: list[int],
        learning_rate: float,
        neuron_threshold_rate: float
    ) -> None:
        self.neurons = neuron_layer
        self.synapses = synapse_layer
        self.neuron_ids = neuron_ids
        self.learning_rate = learning_rate
        self.neuron_threshold_rate = neuron_threshold_rate
        
        self.current_tick: int = 0

        # Буфер для хранения спайков, родившихся НА ПРОШЛОМ ТАКТЕ (T)
        self.previous_tick_spikes: list[tuple[int, float]] = []

    def live_tick(
        self, 
        sensory_input: list[tuple[int, float]],
        reward: float # <-- Добавляем аргумент
    ) -> list[tuple[int, float]]:
        self.current_tick += 1

        # Синапсы переводят спайки прошлого такта в токи текущего и взводят теги
        synaptic_pulses = self.synapses.forward(self.previous_tick_spikes)
        combined_input = synaptic_pulses + sensory_input

        # Нейроны обрабатывают токи и выдают спайки
        current_spikes = self.neurons.forward(combined_input, self.current_tick)

        # --- НОВОЕ ОБУЧЕНИЕ: Трёхфакторный STDP ---
        self.learn_three_factor(reward)

        self.previous_tick_spikes = current_spikes
        return current_spikes

    def learn_three_factor(self, reward: float) -> None:
        """
        Модулируемый Трехфакторный STDP + Активность-зависимый Синаптогенез (Поколение 4.3) + Активность-зависимый Нейрогенез (Поколение 4.4).
        1. Обучает существующие активные синапсы и создает новые нейроны на стресс.
        2. Отращивает новые возбуждающие (+) или тормозные (-) связи при сильном reward/pain.
        """
        # --- ШАГ 1: КЛАССИЧЕСКОЕ ИЗМЕНЕНИЕ СУЩЕСТВУЮЩИХ ВЕСОВ ---
        if abs(reward) >= 0.001:
            for i in list(self.synapses.active_tags):
                if i in self.synapses.free_ids: continue

                tag = self.synapses.synapse_tags[i]
                if tag <= 0.001: continue

                # R-STDP: пересечение локального следа (tag) и глобального фактора среды (reward)
                dw = self.learning_rate * reward * tag

                # Обновляем вес и зажимаем в биологические рамки [-1.0 ... 1.0]
                new_weight = self.synapses.synapse_weights[i] + dw
                self.synapses.synapse_weights[i] = max(-1.0, min(1.0, new_weight))


        # --- ШАГ 2: ДИНАМИЧЕСКИЙ НЕЙРОГЕНЕЗ ПО ОСТРОЙ НУЖДЕ КЛЕТКИ ---
        # Нейрон рождается только при мощнейшем стрессе/кризисе гомеостаза
        NEUROGENESIS_THRESHOLD = 0.5 #FIXME Вынести в характер
        if abs(reward) >= NEUROGENESIS_THRESHOLD:
            # Выращиваем новый интернейрон в физическом слое
            new_neuron_id = self.neurons.add_neuron(
                threshold=self.neuron_threshold_rate,
                decay_rate=self.neurons.decay_rate
            )
            
            # Регистрируем его в логическом скрытом слое Клетки
            self.neuron_ids.append(new_neuron_id)
            
            # Взводим его тег памяти на максимум, чтобы синаптогенез на текущем такте 
            # гарантированно подключил его к горящим сенсорам внешнего мира
            self.neurons.neuron_tags[new_neuron_id] = 1.0
            self.neurons.active_neuron_tags.add(new_neuron_id)


        # --- ШАГ 3: СИНАПТОГЕНЕЗ (РОСТ НОВЫХ СВЕЗЕЙ ПРИ СИЛЬНОМ ОПЫТЕ) ---
        GROWTH_THRESHOLD = 0.15 #FIXME Вынести в характер
        if abs(reward) < GROWTH_THRESHOLD: return

        hot_neurons = list(self.neurons.active_neuron_tags)

        for src_id in hot_neurons:
            for dst_id in hot_neurons:
                if src_id == dst_id: continue

                #FIXME Процесс повторяет логику экспрессии днк - инкапсулировать обе
                synapse_exists = False
                if src_id in self.synapses.source_to_synapses:
                    for syn_idx in self.synapses.source_to_synapses[src_id]:
                        if self.synapses.synapse_targets[syn_idx] == dst_id:
                            synapse_exists = True
                            break

                # Если провода нет — прокладываем его на пересечении следов памяти нейронов!
                if not synapse_exists:
                    # Сила синапса пропорциональна тому, насколько свежи следы памяти обоих нейронов
                    trace_strength = self.neurons.neuron_tags[src_id] * self.neurons.neuron_tags[dst_id]
                    
                    # Награда рождает возбуждение (+), Наказание рождает торможение (-)
                    sign = 1.0 if reward > 0 else -1.0
                    initial_weight = sign * 0.1 * trace_strength
                    
                    self.synapses.add_synapse(
                        source_idx=src_id, 
                        target_idx=dst_id, 
                        weight=initial_weight
                    )



class Cell:
    def __init__(self, config: CellConfig) -> None:
        self.current_time = 0

        self.state = config.state
        self.previous_state = self.state.copy()
        self.homeostasis_receptors = config.homeostasis_weights
        self.buffered_reward = 0.0

        self.character = config.character
        self.sensors = config.sensors
        self.actions = config.actions

        self.all_neurons = NeuronLayer()
        self.all_synapses = SynapseLayer(tag_decay_rate=self.character.tag_decay_rate)

        # Словари для быстрого поиска ID нейронов по их текстовым именам
        self.sensors_map: dict[str, int] = {}
        self.neuron_ids: list[int] = []
        self.actions_map: dict[str, int] = {}

        # Сенсорные нейроны
        for name in self.sensors:
            self.sensors_map[name] = self.all_neurons.add_neuron(
                threshold=self.character.base_neuron_threshold, 
                decay_rate=0.0 # нулевой decay_rate намеренно - сенсоры не затухают
            ) 
        
        # Скрытые интернейроны
        max_hidden_idx = -1
        if hasattr(config, 'dna_synapses') and config.dna_synapses:
            for gene in config.dna_synapses:
                path = gene["path"]
                # Индекс скрытого нейрона всегда находится на позиции [1] в пути
                if len(path) > 2 and isinstance(path[1], int):
                    if path[1] > max_hidden_idx:
                        max_hidden_idx = path[1]

        hidden_neurons_count = max_hidden_idx + 1 if max_hidden_idx >= 0 else 1

        for _ in range(hidden_neurons_count):
            self.neuron_ids.append(
                self.all_neurons.add_neuron(
                    threshold=self.character.base_neuron_threshold, 
                    decay_rate=self.character.decay_rate
                )
            )
        
        # Моторные нейроны
        for name in self.actions:
            self.actions_map[name] = self.all_neurons.add_neuron(
                threshold=self.character.base_neuron_threshold
            )


        # 3. ЭКСПРЕССИЯ ДНК (Прокладка транзитных дорожек)
        if hasattr(config, 'dna_synapses') and config.dna_synapses:
            for gene in config.dna_synapses:
                path = gene["path"]
                weights = gene["weights"]
                
                src_name = path[0]  # Ожидаем строку сенсора
                hidden_idx = path[1] # Ожидаем число, относительный индекс скрытого нейрона
                dst_name = path[2]  # Ожидаем строку мотора
                
                # Валидация: проверяем, существуют ли такие сенсоры и моторы в текущем теле клетки
                if src_name in self.sensors_map and dst_name in self.actions_map:
                    s_id = self.sensors_map[src_name]
                    a_id = self.actions_map[dst_name]
                    
                    # Маппим относительный индекс из ДНК на реальный ID созданного нейрона в скрытом слое
                    if 0 <= hidden_idx < len(self.neuron_ids):
                        h_id = self.neuron_ids[hidden_idx]
                        
                        # Протягиваем физические провода через транзитные слои синапсов
                        w_input = weights[0]
                        w_output = weights[1]
                        
                        # --- ПРОКЛАДКА ВХОДНОГО СИНАПСА ---
                        input_exists = False
                        if s_id in self.all_synapses.source_to_synapses:
                            for syn_idx in self.all_synapses.source_to_synapses[s_id]:
                                if self.all_synapses.synapse_targets[syn_idx] == h_id:
                                    input_exists = True
                                    break
                        
                        if not input_exists:
                            self.all_synapses.add_synapse(source_idx=s_id, target_idx=h_id, weight=w_input)
                        
                        # --- ПРОКЛАДКА ВЫХОДНОГО СИНАПСА (ЗАЩИТА ОТ ДУБЛИКАТОВ) ---
                        # Проверяем, нет ли уже синапса от h_id к a_id (ведь узел 0 используется дважды)
                        output_exists = False
                        if h_id in self.all_synapses.source_to_synapses:
                            for syn_idx in self.all_synapses.source_to_synapses[h_id]:
                                if self.all_synapses.synapse_targets[syn_idx] == a_id:
                                    output_exists = True
                                    break
                        
                        if not output_exists:
                            self.all_synapses.add_synapse(source_idx=h_id, target_idx=a_id, weight=w_output)

        self.snn = SpikingNeuralNetwork(
            neuron_layer=self.all_neurons,
            synapse_layer=self.all_synapses,
            neuron_ids=self.neuron_ids,
            learning_rate=self.character.learning_rate,
            neuron_threshold_rate=self.character.base_neuron_threshold
        )


    def feel(self, sensory_input: dict[str, float]) -> list[tuple[int, float]]:
        """Объединяет внешние раздражители и внутреннее состояние клетки по сквозным ID"""
        output = []

        # 1. Обработка внешних сенсоров
        for name, value in sensory_input.items():
            if name in self.sensors_map:
                global_id = self.sensors_map[name]
                if value > 0.0:
                    output.append((global_id, value))

        # 2. Обработка внутренних потребностей (метаболизм)
        for name, value in self.state.items():
            if name in self.sensors_map:
                global_id = self.sensors_map[name]
                # инвертируем: чем меньше значение ресурса, тем сильнее "голод"
                inverted_value = max(0.0, min(1.0, 1.0 - value))
                if inverted_value > 0.0:
                    output.append((global_id, inverted_value))

        return output


    def choose(
        self, current_network_spikes: list[tuple[int, float]]
    ) -> tuple[str | None, float]:
        """
        Конкурентный выбор действия на основе глобальных спайков сети.
        Разрешает коллизии одновременных спайков через сравнение потенциалов мембраны.
        """
        chosen_action = None
        max_potential = -1.0
        
        # Фильтруем спайки, оставляя только те, которые принадлежат моторным нейронам
        motor_spikes = [
            (name, val) for idx, val in current_network_spikes 
            for name, g_id in self.actions_map.items() if idx == g_id and val > 0
        ]   

        if not motor_spikes: return None, 0.0     
        
        # Если спайков несколько, выбираем тот, у которого сильнее импульс 
        # (в текущей реализации val всегда 1.0, но структура готова к градиенту)
        for name, val in motor_spikes:
            if val > max_potential:
                max_potential = val
                chosen_action = name

        return chosen_action, max_potential


    def act(self, action: str | None):
        """
        Выполнение моторного действия. 
        В реальном мире это заставит клетку сместиться на карте или укусить объект.
        """
        if action is None:
            return

        # Физическое действие стоит АТФ
        self.state["energy"] -= self.character.action_cost

        # ХАРДКОД ДЛЯ ТЕСТА (Имитируем реакцию внешней среды):
        # Если клетка выбрала "eat" на этапе охоты, среда дает ей порцию питательных веществ
        if action == "eat":
            self.state["energy"] += 0.35  # Повышаем энергию (0.35 покроет action_cost и метаболизм)

    #TODO Это первичное подкрепление на основе дельты гомеостаза тела
    # В будущем нужно реализовать дополнительное вторичное подкрепление на основе механизма прогнозирования награды
    def update_metabolic_reward(self) -> None:
        """
        Биологический рецептор дельты. 
        Автоматически считает суммарный сдвиг гомеостаза по всем стейтам.
        """
        total_reward = 0.0

        for state_name, current_val in self.state.items():
            # Если для этого состояния настроен метаболический рецептор
            if state_name in self.homeostasis_receptors:
                prev_val = self.previous_state.get(state_name, current_val)
                delta = current_val - prev_val
                
                receptor = self.homeostasis_receptors[state_name]
                weight = receptor["weight"]
                threshold = receptor["threshold"]

                if abs(delta) < threshold:
                    delta = 0.0

                # Умножаем на биологическую значимость (вес рецептора)
                total_reward += delta * weight

        # Синхронизируем слепок состояний для следующего такта
        self.previous_state = self.state.copy()

        # Зажимаем итоговый дофаминовый/норадреналиновый всплеск в рамки
        self.buffered_reward = max(-1.0, min(1.0, total_reward))


    def live(self, sensory_input: dict[str, float]) -> tuple[str, float]:
        self.current_time += 1

        # ЕДИНЫЙ ТАКТ SNN (Обучение принимает суммарный модулятор)
        sensor_inputs = self.feel(sensory_input)
        network_spikes = self.snn.live_tick(sensor_inputs, reward=self.buffered_reward)
        
        # 3. АТФ за жизнедеятельность и мышление (по количеству спайков такта)
        self.state["energy"] -= self.character.base_metabolism
        self.state["energy"] -= len(network_spikes) * self.character.spike_cost

        # 4. Выбор и выполнение моторного действия
        chosen_action, intensity = self.choose(network_spikes)
        if chosen_action:
            self.act(chosen_action)

        # 5. Ограничение пула стейтов в биологических рамках [0.0 ... 1.0]
        for key in self.state:
            self.state[key] = max(0.0, min(1.0, self.state[key]))

        self.update_metabolic_reward()

        return chosen_action, intensity




def run():
    # Базовое состояние клетки
    cell_state = {
        "energy": 1.0,
        "health": 1.0,
    }
    cell_homeostasis_weights = {
        # Замечаем падение/рост энергии, только если они сильнее фонового метаболизма (~0.01)
        "energy": {"weight": 1.0, "threshold": 0.005}, 
        # Здоровье критично: рецептор улавливает малейшее падение (боль), порог почти нулевой
        "health": {"weight": 1.5, "threshold": 0.001},  
        # Токсины: любая дельта — это повод для паники
        "toxins": {"weight": -1.2, "threshold": 0.0}
    }

    
    # Полный интерфейс ввода (8 сенсоров зрения + 2 сенсора контакта)
    cell_sensors = [
        "light_greenX+", "light_greenX-", "light_greenY+", "light_greenY-", "light_green_inside",
        "light_redX+",   "light_redX-",   "light_redY+",   "light_redY-",   "light_red_inside"
    ]
    
    # Полный интерфейс вывода (6 моторов)
    cell_actions = [
        "eat", "bite", "moveX+", "moveX-", "moveY+", "moveY-"
    ]
    
    # =========================================================================
    # ПРОСТРАНСТВЕННО-КОНТАКТНАЯ ДНК ИНСТИНКТОВ (Поколение 3.4)
    # =========================================================================
    cell_dna_synapses = [
        # --- ТЯГА К ЗЕЛЕНОМУ СВЕТУ (Движение к источнику) ---
        {"path": ["light_greenX+", 0, "moveX+"], "weights": [0.5, 0.7]},
        {"path": ["light_greenX-", 1, "moveX-"], "weights": [0.5, 0.7]},
        {"path": ["light_greenY+", 2, "moveY+"], "weights": [0.5, 0.7]},
        {"path": ["light_greenY-", 3, "moveY-"], "weights": [0.5, 0.7]},
        # КОНТАКТ: Еда прямо под нами -> КУШАТЬ (Хаб 4)
        {"path": ["light_green_inside", 4, "eat"], "weights": [0.8, 0.8]},

        # --- ИЗБЕГАНИЕ КРАСНОГО СВЕТА (Бегство в противоположную сторону) ---
        {"path": ["light_redX+",   5, "moveX-"], "weights": [0.5, 0.7]},
        {"path": ["light_redX-",   6, "moveX+"], "weights": [0.5, 0.7]},
        {"path": ["light_redY+",   7, "moveY-"], "weights": [0.5, 0.7]},
        {"path": ["light_redY-",   8, "moveY+"], "weights": [0.5, 0.7]},
        # КОНТАКТ: Опасность настигла -> КУСАТЬ/АТАКОВАТЬ для защиты (Хаб 9)
        {"path": ["light_red_inside",   9, "bite"], "weights": [0.8, 0.8]}
    ]
    
    # Настройки характера (Пороги унифицированы на 0.8, жесткое затухание 0.30)
    cell_character = CellCharcterConfig(
        learning_rate=0.07,
        decay_rate=0.05,
        tag_decay_rate=0.05,
        base_metabolism=0.002,
        base_neuron_threshold=0.8,
        spike_cost=0.0005,
        action_cost=0.01
    )

    # Инициализация (Блок динамического нейрогенеза сам вырастит 10 скрытых нейронов)
    cell = Cell(CellConfig(
        state=cell_state,
        homeostasis_weights=cell_homeostasis_weights,
        sensors=cell_sensors, 
        actions=cell_actions,
        character=cell_character,
        dna_synapses=cell_dna_synapses
    ))

    empty_input = {s: 0.0 for s in cell_sensors}

    print("\n=== ЭТАП 1: СЦЕНАРИЙ ОХОТЫ НА ЕДУ ПО ОСИ X+ (6 тактов) ===")
    print("Такты 1-3: Еда далеко справа (Ждем: moveX+). Такт 4-6: Еда прямо под нами (Ждем: переключение на eat)")
    for t in range(20):
        sensory = empty_input.copy()
        if t < 15:
            sensory["light_greenX+"] = 1.0       # Видим справа, бежим
        else:
            sensory["light_green_inside"] = 1.0   # Догнали, кушаем!
            
        action, potential = cell.live(sensory)
        print(f"Такт {cell.current_time} | Охота | Выбрано: {action}")

    print("\n=== ЭТАП 2: СЦЕНАРИЙ ОБОРОНЫ ОТ УГРОЗЫ ПО ОСИ Y- (6 тактов) ===")
    print("Такты 7-9: Опасность снизу (Ждем: убегаем вверх -> moveY+). Такт 10-12: Опасность зажала в угол (inside) (Ждем: отбиваемся -> bite)")
    for t in range(20):
        sensory = empty_input.copy()
        if t < 15:
            sensory["light_redY-"] = 1.0         # Опасность снизу, бежим вверх
        else:
            sensory["light_red_inside"] = 1.0     # Опасность настигла, кусаем!
            
        action, potential = cell.live(sensory)
        print(f"Такт {cell.current_time} | Оборона | Выбрано: {action}")
