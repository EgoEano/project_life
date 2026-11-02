from dataclasses import dataclass, field

# Тестовая заготовка
@dataclass
class Synapse:
    weight: float


# индекс нейрона и его значение
# dict[int, float]
# индекс нейрона в индексе синапсов с его значением
# dict[int, dict[int, float]]
class SynapticProjection:
    def __init__(
        self, 
        synapses: dict[int, dict[int, float]] | None = None
    ) -> None:
        self.synapses: dict[int, dict[int, float]] = {
            src: dict(dsts) for src, dsts in synapses.items()
            } if synapses else {}


    def forward(
        self, 
        input: dict[int, float], 
    ) -> dict[int, float]:

        output: dict[int, float] = {}

        for src, activity in input.items():
            if activity == 0: continue

            for dst, weight in self.synapses.get(src, {}).items():
                output[dst] = output.get(dst, 0.0) + (activity * weight)

        return output


    # Не проверено, поэтому не пускаем в работу
    # Возможно leard уйдет в другое место отработки синаптической пластичности
    # def learn(self, source_activity: dict[int, float], target_activity: dict[int, float]) -> None:
    #     # Шаг 1: Сначала плавно гасим ВСЕ существующие синапсы от времени (фоновое забывание)
    #     for src in list(self.synapses.keys()):
    #         for dst in list(self.synapses[src].keys()):
    #             # hebb_decay
    #             self.synapses[src][dst] -= self.character.decay_rate * self.synapses[src][dst]
                
    #             # Если синапс полностью испарился — чистим ОЗУ
    #             if self.synapses[src][dst] <= 0:
    #                 del self.synapses[src][dst]
    #         if not self.synapses[src]:
    #             del self.synapses[src]

    #     # Шаг 2: Укрепляем синапсы только там, где нейроны реально горят одновременно
    #     for src, act_src in source_activity.items():
    #         for dst, act_dst in target_activity.items():
    #             if act_src > 0 and act_dst > 0:
    #                 hebb_change = self.character.learning_rate * (act_src * act_dst)
                    
    #                 if src not in self.synapses:
    #                     self.synapses[src] = {}

    #                 current_weight = self.synapses.get(src, {}).get(dst, 0.0)
    #                 hebb_change = self.character.learning_rate * (act_src * act_dst)
    #                 new_weight = current_weight + hebb_change
    #                 self.synapses[src][dst] = max(-1.0, min(1.0, new_weight))



class SparseBioLayer:
    def __init__(self) -> None:
        # Храним данные лениво: только для тех, кто не в базовом состоянии
        self.potentials: dict[int, float] = {}
        self.thresholds: dict[int, float] = {}
        
        # Константы (параметры гомеостаза)
        self.base_threshold = 1.0
        self.leak_rate = 0.2
        self.threshold_step = 0.1  # Прирост порога при спайке
        self.threshold_decay = 0.01 # Скорость остывания порога

    def process_input(self, accumulated_current: dict[int, float]) -> dict[int, float]:
        active_outputs: dict[int, float] = {}
        
        # Шаг 1: Обрабатываем только те нейроны, куда пришел ток
        for idx, new_input in accumulated_current.items():
            if new_input <= 0: continue
            
            # Считаем заряд с учетом утечки
            old_charge = self.potentials.get(idx, 0.0)
            current_potential = (old_charge * (1.0 - self.leak_rate)) + new_input
            
            # Получаем текущий порог
            threshold = self.thresholds.get(idx, self.base_threshold)

            # Проверка на спайк
            if current_potential >= threshold:
                active_outputs[idx] = current_potential 
                self.potentials[idx] = 0.0  # Сброс заряда
                self.thresholds[idx] = threshold + self.threshold_step  # Защита от эпилепсии
            else:
                self.potentials[idx] = current_potential

        # Шаг 2: Ленивое «остывание» порогов и утечка неактивных нейронов
        # Остывать должны только те, у кого порог выше базового
        for idx in list(self.thresholds.keys()):
            if idx in active_outputs: continue # Пропускаем тех, кто только что выстрелил
            
            new_th = self.thresholds[idx] - self.threshold_decay
            if new_th <= self.base_threshold:
                del self.thresholds[idx] # Полностью очищаем память ОЗУ
            else:
                self.thresholds[idx] = new_th

        # То же самое для остаточных потенциалов (если утекли в ноль — удаляем)
        for idx in list(self.potentials.keys()):
            if idx in accumulated_current: continue # Мы их уже обработали выше
            
            new_pot = self.potentials[idx] * (1.0 - self.leak_rate)
            if new_pot < 0.01:
                del self.potentials[idx] # Очистка памяти ОЗУ
            else:
                self.potentials[idx] = new_pot

        return active_outputs