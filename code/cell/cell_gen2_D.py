from dataclasses import dataclass, field


@dataclass
class CellCharcterConfig:
    learning_rate: float
    decay_rate: float

@dataclass
class CellConfig:
    state: dict[str, float]
    character: CellCharcterConfig

    sensors: list[str]
    concepts: list[str]
    actions: list[str]

    dna_sensory_to_concept: dict[str, dict[str, float]] = field(default_factory=dict)
    dna_internal_to_concept: dict[str, dict[str, float]] = field(default_factory=dict)
    dna_concept_to_motor: dict[str, dict[str, float]] = field(default_factory=dict)
    dna_motor_to_concept: dict[str, dict[str, float]] = field(default_factory=dict)


class SynapticProjection:
    def __init__(
        self, 
        character: CellCharcterConfig,
        dna_synapses: dict[str, dict[str, float]] = None
    ) -> None:
        self.character = character
        self.synapses = {src: dict(dsts) for src, dsts in dna_synapses.items()} if dna_synapses else {}


    def forward(self, source_activity: dict[str, float], target_activity: dict[str, float]) -> dict[str, float]:
        for src, current_signal in source_activity.items():
            if current_signal <= 0: continue

            connections = self.synapses.get(src, {})
            for dst, channel_width in connections.items():
                # channel_width может ли быть отрицательным (торможение)?
                target_activity[dst] = target_activity.get(dst, 0.0) + (current_signal * channel_width)

        # зачем здесь через лист?
        # for name in list(target_activity.keys()):
        #     target_activity[name] = max(-1.0, min(1.0, target_activity[name]))

        #     if target_activity[name] == 0:
        #             del target_activity[name]

        return target_activity


    def learn(self, source_activity: dict[str, float], target_activity: dict[str, float]) -> None:
        # Шаг 1: Сначала плавно гасим ВСЕ существующие синапсы от времени (фоновое забывание)
        for src in list(self.synapses.keys()):
            for dst in list(self.synapses[src].keys()):
                # hebb_decay
                self.synapses[src][dst] -= self.character.decay_rate * self.synapses[src][dst]
                
                # Если синапс полностью испарился — чистим ОЗУ
                if self.synapses[src][dst] <= 0:
                    del self.synapses[src][dst]
            if not self.synapses[src]:
                del self.synapses[src]

        # Шаг 2: Укрепляем синапсы только там, где нейроны реально горят одновременно
        for src, act_src in source_activity.items():
            for dst, act_dst in target_activity.items():
                if act_src > 0 and act_dst > 0:
                    hebb_change = self.character.learning_rate * (act_src * act_dst)
                    
                    if src not in self.synapses:
                        self.synapses[src] = {}

                    current_weight = self.synapses.get(src, {}).get(dst, 0.0)
                    hebb_change = self.character.learning_rate * (act_src * act_dst)
                    new_weight = current_weight + hebb_change
                    self.synapses[src][dst] = max(-1.0, min(1.0, new_weight))

    # def learn(
    #     self, 
    #     source_activity: dict[str, float], 
    #     target_activity: dict[str, float]
    # ) -> None:
    #     for src, act_src in source_activity.items():
    #         for dst, act_dst in target_activity.items():
    #             if act_src <= 0 or act_dst <= 0: continue

    #             current_weight = self.synapses.get(src, {}).get(dst, 0.0)
    #             hebb_change = self.character.learning_rate * (act_src * act_dst)
    #             hebb_decay = self.character.decay_rate * current_weight

    #             new_weight = current_weight + (hebb_change - hebb_decay)

    #             if new_weight > 0:
    #                 if src not in self.synapses:
    #                     self.synapses[src] = {}
    #                 self.synapses[src][dst] = max(-1.0, min(1.0, new_weight))
    #             elif src in self.synapses and dst in self.synapses[src]:
    #                 del self.synapses[src][dst]



class Cell:
    def __init__(self, config: CellConfig) -> None:
        self.state = config.state
        self.character = config.character
        
        self.sensors = config.sensors
        self.concepts = config.concepts
        self.actions = config.actions

        self.sensory_to_concept = SynapticProjection(self.character, config.dna_sensory_to_concept)
        self.internal_to_concept = SynapticProjection(self.character, config.dna_internal_to_concept) 
        self.concept_to_motor = SynapticProjection(self.character, config.dna_concept_to_motor)
        self.motor_to_concept = SynapticProjection(self.character, config.dna_motor_to_concept)        # neurons and synapses with DNA preset

        self.sensor_activity = {}
        self.internal_activity = {}
        self.concept_activity = {}
        self.action_activity = {}


    def focus(self):
        self.sensor_activity = {}
        self.internal_activity = {}
        self.concept_activity = {}
        self.action_activity = {}


    def feel(self, sensory_input: dict[str, float]):
        for name in self.sensors:
            val = sensory_input.get(name, 0.0)
            if val > 0:
                self.sensor_activity[name] = val

        # такое ощущение что здесь неверно - внутренние сенсоры словно как то грязно смешиваются с обычными, а реверсирование значений тоже выглядит странно - словно бы сеть должна оперировать прямыми показателями корректно
        for name in self.concepts:
            val = self.state[name]
            if val > 0:
                self.internal_activity[name] = val


    def react(self):
        self.concept_activity = self.sensory_to_concept.forward(self.sensor_activity, self.concept_activity)
        self.concept_activity = self.internal_to_concept.forward(self.internal_activity, self.concept_activity)

        for name in list(self.concept_activity.keys()):
            self.concept_activity[name] = max(0.0, min(1.0, self.concept_activity[name]))
            if self.concept_activity[name] == 0:
                del self.concept_activity[name]

        self.action_activity = self.concept_to_motor.forward(self.concept_activity, self.action_activity)

        for name in list(self.action_activity.keys()):
            self.action_activity[name] = max(0.0, min(1.0, self.action_activity[name]))
            if self.action_activity[name] == 0:
                del self.action_activity[name]


    def learn(self):
        # Полагаю петлю обратной связи обучения можно просто провести до конца наверх, закончив "эхо опыта" и на сенсорах
        self.sensory_to_concept.learn(self.sensor_activity, self.concept_activity)
        self.internal_to_concept.learn(self.internal_activity, self.concept_activity)
        self.concept_to_motor.learn(self.concept_activity, self.action_activity)
        self.motor_to_concept.learn(self.action_activity, self.concept_activity)

    def act(self):
        chosen_action = max(self.actions, key=lambda idx: self.action_activity.get(idx, 0.0))
        action_power = self.action_activity.get(chosen_action, 0.0)
        return chosen_action, action_power

    def live(self, sensory_input: dict[str, float]):
        self.focus()
        self.feel(sensory_input)
        self.react()
        self.learn()
        return self.act()



def run():
    # 1. СТЫКОВКА ДНК: Все имена должны строго соответствовать анатомическому атласу!
    born_dna_sensory_to_concept = {
        "color_green": {"energy": 0.8}  # Зеленый свет активирует концепт "energy" ума
    }
    born_dna_internal_to_concept = {
        "energy": {"energy": -0.5}      # Прямая сырость тела тормозит этот же концепт "energy" ума
    }
    born_dna_concept_to_motor = {
        "energy": {"bite": 0.7}         # Концепт "energy" ума толкает ток в физический мотор укуса
    }

    # 2. Собираем полную конфигурацию
    config = CellConfig(
        state={"energy": 0.2}, # Клетка голодна (0.2 из 1.0)
        character=CellCharcterConfig(learning_rate=0.3, decay_rate=0.01),

        # Анатомический атлас:
        sensors=["color_green", "color_red"], # Внешние провода
        concepts=["energy"],                  # Внутренний провод от тела + имя концепта в уме
        actions=["bite", "move_away"],        # Выходные провода

        dna_sensory_to_concept=born_dna_sensory_to_concept,
        dna_internal_to_concept=born_dna_internal_to_concept,
        dna_concept_to_motor=born_dna_concept_to_motor,
        dna_motor_to_concept={} 
    )

    cell = Cell(config)

    # 3. ЦИКЛ ТЕСТИРОВАНИЯ
    print("🤖 Запуск симуляции. Клетка голодна (энергия 0.2)...")
    for i in range(5):
        # ПЕРЕДАЕМ ПРАВИЛЬНЫЙ КЛЮЧ: "color_green" вместо "color", чтобы провод состыковался!
        act, power = cell.live({ "color_green": 1.0 })
        print(f"Такт {i+1}: Действие = {act}, Сила = {power:.4f}")

    # Имитируем, что клетка наелась (энергия выросла до максимума)
    cell.state["energy"] = 1.0
    print("\n🍔 Клетка полностью наелась (энергия 1.0)...")

    for i in range(5):
        act, power = cell.live({ "color_green": 0.0 })
        print(f"Такт {i+6}: Действие = {act}, Сила = {power:.4f}")
