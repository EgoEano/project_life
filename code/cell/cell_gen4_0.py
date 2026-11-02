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
        
        self.neuron_decays = [decay_rate] * len(self.neuron_potentials)

        self.free_ids: set[int] = set()

  
    def add_neuron(self, threshold: float = 1.0, decay_rate: float = 0.05) -> int:
        if self.free_ids:
            new_neuron_id = self.free_ids.pop()

            self.neuron_potentials[new_neuron_id] = 0.0
            self.neuron_thresholds[new_neuron_id] = threshold
            self.neuron_timestamps[new_neuron_id] = 0
            self.neuron_decays[new_neuron_id] = decay_rate

            return new_neuron_id
        
        self.neuron_potentials.append(0.0)
        self.neuron_thresholds.append(threshold)
        self.neuron_timestamps.append(0)
        self.neuron_decays.append(decay_rate)

        return len(self.neuron_potentials) - 1


    def remove_neuron(self, neuron_id: int):
        if 0 <= neuron_id < len(self.neuron_potentials):
            self.neuron_potentials[neuron_id] = 0.0
            self.neuron_thresholds[neuron_id] = 0.0
            self.neuron_decays[neuron_id] = 0.0
            self.neuron_timestamps[neuron_id] = 0
            self.free_ids.add(neuron_id)


    def forward(
        self, 
        input: list[tuple[int, float]], # id: [-1...1],
        current_time: int
    ) -> list[tuple[int, float]]: # id: [0...1]

        output: list[tuple[int, float]] = []

        # 1. new inputs
        for idx, weight in input:
            if idx >= len(self.neuron_potentials) or idx in self.free_ids: continue

            dt = current_time - self.neuron_timestamps[idx]
            if dt > 0:
                # Математически точное экспоненциальное затухание за все пропущенные такты
                self.neuron_potentials[idx] *= math.pow(1.0 - self.neuron_decays[idx], dt)

            self.neuron_potentials[idx] += weight
            self.neuron_timestamps[idx] = current_time

        # 2. checking thresholds and generating spikes
        for idx in range(len(self.neuron_potentials)):
            if idx in self.free_ids: continue

            self.all_neurons_or_self_timestamps(idx, current_time)

            if self.neuron_potentials[idx] >= self.neuron_thresholds[idx]:
                output.append((idx, 1.0))

                self.neuron_potentials[idx] -= self.neuron_thresholds[idx]
                
                if self.neuron_potentials[idx] >= self.neuron_thresholds[idx]:
                    self.neuron_potentials[idx] = 0.0

                self.neuron_timestamps[idx] = current_time

        return output


    def all_neurons_or_self_timestamps(self, idx: int, current_time: int) -> int:
        """Вспомогательный метод ленивого leak-обновления перед чтением состояния нейрона"""
        dt = current_time - self.neuron_timestamps[idx]
        if dt > 0:
            self.neuron_potentials[idx] *= math.pow(1.0 - self.neuron_decays[idx], dt)
            self.neuron_timestamps[idx] = current_time
        return current_time


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

        self.synapse_tags: list[float] = [] 
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
            self.synapse_tags[idx] = 0.0

            # Регистрируем синапс в индексной карте
            self.source_to_synapses[source_idx].append(idx)

            return idx

        self.synapse_sources.append(source_idx)
        self.synapse_targets.append(target_idx)
        self.synapse_weights.append(weight)
        self.synapse_tags.append(0.0)

        idx = len(self.synapse_weights) - 1
        self.source_to_synapses[source_idx].append(idx)

        return idx


    def remove_synapse(self, synapse_id: int):
        if 0 <= synapse_id < len(self.synapse_weights):
            src = self.synapse_sources[synapse_id]
            if synapse_id in self.source_to_synapses[src]:
                self.source_to_synapses[src].remove(synapse_id)

            self.synapse_weights[synapse_id] = 0.0
            self.synapse_tags[synapse_id] = 0.0
            
            self.free_ids.add(synapse_id)


    def forward(
        self, 
        input: list[tuple[int, float]]
    ) -> list[tuple[int, float]]:
        # --- ПОКОЛЕНИЕ 4.0: ЕЖЕТАКТОВОЕ ОСТЫВАНИЕ СЛЕДОВ ПАМЯТИ ---
        # Время идет: все теги в сети плавно угасают на 20% за такт
        for i in range(len(self.synapse_tags)):
            if i in self.free_ids: continue
            self.synapse_tags[i] *= 0.8

        active_dict = {neuron_id: spike_val for neuron_id, spike_val in input if spike_val > 0}

        if not active_dict: return []

        # Словарь для ОПТИМИЗАЦИИ 2: Агрегируем заряды по целевым нейронам { ID_цели: суммарный_заряд }
        aggregated_outputs: dict[int, float] = defaultdict(float)

        # EVENT-DRIVEN ПРОХОД: Перебираем только те нейроны, которые Реально Выдали Спайк
        for src_id, spike_val in active_dict.items():
            # Если для этого нейрона нет синапсов, просто идем дальше
            if src_id not in self.source_to_synapses: continue
                
            # Достаем только связанные с ним синапсы
            for syn_idx in self.source_to_synapses[src_id]:
                # Замечание 1: Больше не проверяем "weight == 0.0", так как синапс может ожить при STDP. 
                # Проверяем только то, что синапс физически не удален (нет в free_ids).
                if syn_idx in self.free_ids: continue
                    

                # --- ПОКОЛЕНИЕ 4.0: ВЗВОДИМ СИНАПТИЧЕСКИЙ ТЕГ ---
                self.synapse_tags[syn_idx] = 1.0

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
        synapse_layer: SynapseLayer,
        learning_rate: float
    ) -> None:
        self.neurons = neuron_layer
        self.synapses = synapse_layer
        self.learning_rate = learning_rate
        
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
        """
        # 1: Инкрементируем глобальное время (ровно один раз за такт)
        self.current_tick += 1

        # 2: Синапсы берут спайки из ПРОШЛОГО такта и превращают их в импульсы для текущего.
        # Это обеспечивает Propagation Delay (синаптическую задержку).
        synaptic_pulses = self.synapses.forward(self.previous_tick_spikes)

        # 3. Подмешиваем свежий сенсорный ввод к синаптическим токам
        combined_input = synaptic_pulses + sensory_input

        # 4: Единственный проход по слою нейронов за такт.
        # Нейроны применяют leak (затухание), суммируют combined_input и генерируют новые спайки.
        current_spikes = self.neurons.forward(combined_input, self.current_tick)

        # 5: Выделяем сквозной набор ID активных нейронов для обучения
        active_neurons = {idx for idx, val in current_spikes if val > 0}

        # Автоматическое обучение на основе произошедшего такта
        self.learn_active_synapses(active_neurons)

        # 5: Сохраняем сгенерированные спайки. 
        # Синапсы обработают их на следующем такте (Т + 1).
        self.previous_tick_spikes = current_spikes

        # Шаг 6: Возвращаем спайки текущего такта наружу.
        # Внешний класс (Cell) сам отфильтрует их по actions_map, чтобы выполнить моторные действия.
        return current_spikes

    def learn_active_synapses(self, active_neuron_ids: set[int]):
        """
        Встроенный автоматический STDP.
        Проверяет сквозной пул синапсов и обучает те связи, где была активность.
        """
        tau = 3.0
        
        for i in range(len(self.synapses.synapse_weights)):
            if i in self.synapses.free_ids:
                continue
                
            src_id = self.synapses.synapse_sources[i]
            dst_id = self.synapses.synapse_targets[i]
            
            # Обучаем синапс только если хотя бы пре- или пост-нейрон сработал СЕЙЧАС
            if src_id not in active_neuron_ids and dst_id not in active_neuron_ids:
                continue
                
            t_src = self.neurons.neuron_timestamps[src_id]
            t_dst = self.neurons.neuron_timestamps[dst_id]
            
            if t_src < 0 or t_dst < 0:
                continue
                
            dt = t_dst - t_src
            
            if abs(dt) > tau * 2: 
                continue

            if dt > 0:
                dw = self.learning_rate * math.exp(-dt / tau)
            elif dt < 0:
                dw = -self.learning_rate * math.exp(dt / tau)
            else:
                dw = self.learning_rate * 0.5
                
            new_weight = self.synapses.synapse_weights[i] + dw
            self.synapses.synapse_weights[i] = max(-1.0, min(1.0, new_weight))



class Cell:
    def __init__(self, config: CellConfig) -> None:
        self.current_time = 0

        self.state = config.state
        self.character = config.character
        self.sensors = config.sensors
        self.actions = config.actions

        self.all_neurons = NeuronLayer()
        self.all_synapses = SynapseLayer()

        # Словари для быстрого поиска ID нейронов по их текстовым именам
        self.sensors_map: dict[str, int] = {}
        self.erste_ids: list[int] = []
        self.actions_map: dict[str, int] = {}

        # Сенсорные нейроны
        for name in self.sensors:
            self.sensors_map[name] = self.all_neurons.add_neuron(threshold=0.8, decay_rate=0.0)    
        
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
            self.erste_ids.append(self.all_neurons.add_neuron(threshold=0.8, decay_rate=self.character.decay_rate))
        
        # Моторные нейроны
        for name in self.actions:
            self.actions_map[name] = self.all_neurons.add_neuron(threshold=0.8)


        # 3. ЭКСПРЕССИЯ ДНК (Прокладка транзитных дорожек)
        if hasattr(config, 'dna_synapses') and config.dna_synapses:
            for gene in config.dna_synapses:
                path = gene["path"]
                weights = gene["weights"]
                
                src_name = path[0]  # Ожидаем строку, например "energy"
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
            learning_rate=self.character.learning_rate
        )


    def feel(self, sensory_input: dict[str, float]) -> list[tuple[int, float]]:
        """Объединяет внешние раздражители и внутреннее состояние клетки по сквозным ID"""
        output = []

        # 1. Обработка внешних сенсоров
        for name, value in sensory_input.items():
            if name in self.sensors_map:
                global_id = self.sensors_map[name]
                if value >= 0.5:
                    output.append((global_id, value))

        # 2. Обработка внутренних потребностей (метаболизм)
        for name, value in self.state.items():
            if name in self.sensors_map:
                global_id = self.sensors_map[name]
                # Например, инвертируем: чем меньше значение ресурса, тем сильнее "голод"
                inverted_value = max(0.0, min(1.0, 1.0 - value))
                # Зачем такой порог? Зачем отсекать сигнал?
                # Кажется была проблема с шумом у ранних поколений - протестировать как сейчас
                if inverted_value >= 0.5:
                    output.append((global_id, inverted_value))

        return output


    def choose(self, current_network_spikes: list[tuple[int, float]]) -> tuple[str | None, float]:
        """
        Конкурентный выбор действия на основе глобальных спайков сети.
        Разрешает коллизии одновременных спайков через сравнение потенциалов мембраны.
        """
        chosen_action = None
        max_potential = -1.0
        
        # Выделяем множество глобальных ID, которые Реально Спайканули на этом такте
        active_global_ids = {idx for idx, val in current_network_spikes if val > 0}
        
        # --- КРИТИЧЕСКИЙ ФИКС ЛЕНИВОГО ЗАТУХАНИЯ ---
        # Перед опросом моторов, принудительно подтягиваем их потенциалы 
        # к текущему времени симуляции, чтобы учесть накопленный decay в простое
        for global_id in self.actions_map.values():
            self.all_neurons.all_neurons_or_self_timestamps(global_id, self.current_time)

        # 1. Если есть активные спайки, выбираем СРЕДИ НИХ тот, у кого выше мембранный потенциал
        if active_global_ids:
            for name, global_id in self.actions_map.items():
                if global_id in active_global_ids:
                    # Проверяем потенциал нейрона (включая избыточный остаток до вычитания или после)
                    pot = self.all_neurons.neuron_potentials[global_id]
                    if pot > max_potential:
                        max_potential = pot
                        chosen_action = name
            # Так как это был реальный спайк, силу активации для внешней среды ставим на максимум
            return chosen_action, 1.0
            
        # 2. Если реальных спайков нет вообще, выбираем по наивысшему накопленному подпороговому потенциалу
        else:
            for name, global_id in self.actions_map.items():
                pot = self.all_neurons.neuron_potentials[global_id]
                if pot > max_potential:
                    max_potential = pot
                    chosen_action = name

        return chosen_action, max_potential


    def act():
        """
        Поместить сюда последствия выбора: смена гомеостаза, состояний и прочего
        """
        pass


    def live(self, sensory_input: dict[str, float]) -> tuple[str, float]:
        self.current_time += 1
        
        # 1. Списание базового метаболизма (простой тоже стоит АТФ)
        # Возможно переместится в act, но это не точно
        self.state["energy"] -= self.character.base_metabolism

        # 2. Считываем датчики и генерируем спайки на сенсорном слое
        sensor_inputs = self.feel(sensory_input)

        # 3. ЕДИНЫЙ ТАКТ SNN (Просчет токов, задержки в 1 такт и авто-обучение STDP)
        network_spikes = self.snn.live_tick(sensor_inputs)
        
        # --- ИНТЕГРАЦИЯ АТФ ЗА МЫШЛЕНИЕ ---
        # Считаем суммарное количество спайков во всей сети за этот такт
        total_spikes = len(network_spikes)
        self.state["energy"] -= total_spikes * self.character.spike_cost

        # 4. Выбор действия на основе того, у какого моторного нейрона сейчас наивысший потенциал
        chosen_action, max_potential = self.choose(network_spikes)

        # 5. Ограничение пула энергии в биологических рамках [0.0 ... 1.0]
        self.state["energy"] = max(0.0, min(1.0, self.state["energy"]))

        return chosen_action, max_potential




def run():
    # Базовое состояние клетки
    cell_state = {
        "energy": 1.0,
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
        {"path": ["light_greenX+", 0, "moveX+"], "weights": [0.8, 0.8]},
        {"path": ["light_greenX-", 1, "moveX-"], "weights": [0.8, 0.8]},
        {"path": ["light_greenY+", 2, "moveY+"], "weights": [0.8, 0.8]},
        {"path": ["light_greenY-", 3, "moveY-"], "weights": [0.8, 0.8]},
        # КОНТАКТ: Еда прямо под нами -> КУШАТЬ (Хаб 4)
        {"path": ["light_green_inside", 4, "eat"], "weights": [0.8, 0.8]},

        # --- ИЗБЕГАНИЕ КРАСНОГО СВЕТА (Бегство в противоположную сторону) ---
        {"path": ["light_redX+",   5, "moveX-"], "weights": [0.8, 0.8]},
        {"path": ["light_redX-",   6, "moveX+"], "weights": [0.8, 0.8]},
        {"path": ["light_redY+",   7, "moveY-"], "weights": [0.8, 0.8]},
        {"path": ["light_redY-",   8, "moveY+"], "weights": [0.8, 0.8]},
        # КОНТАКТ: Опасность настигла -> КУСАТЬ/АТАКОВАТЬ для защиты (Хаб 9)
        {"path": ["light_red_inside",   9, "bite"], "weights": [0.8, 0.8]}
    ]
    
    # Настройки характера (Пороги унифицированы на 0.8, жесткое затухание 0.30)
    cell_character = CellCharcterConfig(
        learning_rate=0.3,
        decay_rate=0.30,        
        base_metabolism=0.01,
        spike_cost=0.001,
        action_cost=0.05
    )

    # Инициализация (Блок динамического нейрогенеза сам вырастит 10 скрытых нейронов)
    cell = Cell(CellConfig(
        state=cell_state,
        sensors=cell_sensors, 
        actions=cell_actions,
        character=cell_character,
        dna_synapses=cell_dna_synapses
    ))

    def print_current_network_weights():
        inv_sensors = {v: k for k, v in cell.sensors_map.items()}
        inv_actions = {v: k for k, v in cell.actions_map.items()}
        print("\n--- Текущая топология синаптических связей ---")
        for i in range(len(cell.all_synapses.synapse_weights)):
            if i in cell.all_synapses.free_ids: continue
            src = cell.all_synapses.synapse_sources[i]
            dst = cell.all_synapses.synapse_targets[i]
            w = cell.all_synapses.synapse_weights[i]
            
            src_name = inv_sensors[src] if src in inv_sensors else (f"Скрытый_{cell.erste_ids.index(src)}" if src in cell.erste_ids else f"ID_{src}")
            dst_name = inv_actions[dst] if dst in inv_actions else (f"Скрытый_{cell.erste_ids.index(dst)}" if dst in cell.erste_ids else f"ID_{dst}")
            print(f"  Синапс [{src_name} -> {dst_name}]: вес = {w:.2f}")

    print("=== СТАРТОВЫЕ ВЕСА СИНАПСОВ ===")
    print_current_network_weights()

    empty_input = {s: 0.0 for s in cell_sensors}

    print("\n=== ЭТАП 1: СЦЕНАРИЙ ОХОТЫ НА ЕДУ ПО ОСИ X+ (6 тактов) ===")
    print("Такты 1-3: Еда далеко справа (Ждем: moveX+). Такт 4-6: Еда прямо под нами (Ждем: переключение на eat)")
    for t in range(6):
        sensory = empty_input.copy()
        if t < 3:
            sensory["light_greenX+"] = 1.0       # Видим справа, бежим
        else:
            sensory["light_green_inside"] = 1.0   # Догнали, кушаем!
            
        action, potential = cell.live(sensory)
        print(f"Такт {cell.current_time} | Охота | Выбрано: {action}")

    print("\n=== ЭТАП 2: СЦЕНАРИЙ ОБОРОНЫ ОТ УГРОЗЫ ПО ОСИ Y- (6 тактов) ===")
    print("Такты 7-9: Опасность снизу (Ждем: убегаем вверх -> moveY+). Такт 10-12: Опасность зажала в угол (inside) (Ждем: отбиваемся -> bite)")
    for t in range(6):
        sensory = empty_input.copy()
        if t < 3:
            sensory["light_redY-"] = 1.0         # Опасность снизу, бежим вверх
        else:
            sensory["light_red_inside"] = 1.0     # Опасность настигла, кусаем!
            
        action, potential = cell.live(sensory)
        print(f"Такт {cell.current_time} | Оборона | Выбрано: {action}")
