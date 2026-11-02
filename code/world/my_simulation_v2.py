import math
import random

from cell.main import makeCell
from world.simulation_world import SimulationWorld

class MySimulationV2(SimulationWorld):
    def __init__(self):
        super().__init__(1200, 600)
        
        self.cell = makeCell()

        self.cell_x = 500
        self.cell_y = 500
        self.cell_size = 16
        # Рисуем клетку на холсте (голубой кружок)
        self.cell_graphic_id = self.canvas.create_oval(
            self.cell_x, self.cell_y, 
            self.cell_x + self.cell_size, self.cell_y + self.cell_size, 
            fill="cyan", outline=""
        )

        # 2. МЕХАНИКА МЫШКИ (Для спавна тестовых объектов)
        self.mouse_objects = {1: None, 2: None, 3: None}
        
        # ЛКМ - Зеленая Еда
        self.canvas.bind("<Button-1>", lambda e: self.on_mouse_press(e, 1, "green", "food"))
        self.canvas.bind("<B1-Motion>", lambda e: self.on_mouse_drag(e, 1))
        self.canvas.bind("<ButtonRelease-1>", lambda e: self.on_mouse_release(e, 1))
        
        # Нажатие на колесико мыши (ККМ) — Синяя неизвестная угроза
        self.canvas.bind("<Button-2>", lambda e: self.on_mouse_press(e, 2, "blue", "unknown_hazard"))
        self.canvas.bind("<B2-Motion>", lambda e: self.on_mouse_drag(e, 2))
        self.canvas.bind("<ButtonRelease-2>", lambda e: self.on_mouse_release(e, 2))

        # ПКМ - Красный Огонь/Угроза
        self.canvas.bind("<Button-3>", lambda e: self.on_mouse_press(e, 3, "red", "fire"))
        self.canvas.bind("<B3-Motion>", lambda e: self.on_mouse_drag(e, 3))
        self.canvas.bind("<ButtonRelease-3>", lambda e: self.on_mouse_release(e, 3))


    def on_mouse_press(self, event, button: int, color: str, obj_type: str):
        """Срабатывает при первом нажатии кнопки мыши"""
        # Если объект для этой кнопки уже почему-то существует, удаляем его
        if self.mouse_objects[button] is not None:
            self.remove_object(self.mouse_objects[button])
            
        # Спавним объект прямо в координатах курсора (центрируем с учетом размера)
        size = 14 if button == 3 else 10  # Сделаем красный чуть крупнее
        x = event.x - size // 2
        y = event.y - size // 2
        
        # Сохраняем сгенерированный uuid
        self.mouse_objects[button] = self.spawn_object(obj_type, x, y, size=size, color=color)

    def on_mouse_drag(self, event, button: int):
        """Срабатывает при движении мыши с зажатой кнопкой"""
        obj_id = self.mouse_objects[button]
        if obj_id is None or obj_id not in self.entities:
            return
            
        obj = self.entities[obj_id]
        
        # Вычисляем разницу между текущим положением курсора и центром объекта
        current_center_x = obj["x"] + obj["size"] // 2
        current_center_y = obj["y"] + obj["size"] // 2
        
        dx = event.x - current_center_x
        dy = event.y - current_center_y
        
        # Двигаем объект на эту дельту
        self.move_object(obj_id, dx, dy)

    def on_mouse_release(self, event, button: int):
        """Срабатывает, когда кнопка мыши отпущена"""
        obj_id = self.mouse_objects[button]
        if obj_id is not None:
            self.remove_object(obj_id)
            self.mouse_objects[button] = None


    def print_current_network_weights(self):
        inv_sensors = {v: k for k, v in self.cell.sensors_map.items()}
        inv_actions = {v: k for k, v in self.cell.actions_map.items()}
        print("\n--- Текущая топология синаптических связей ---")
        for i in range(len(self.cell.all_synapses.synapse_weights)):
            if i in self.cell.all_synapses.free_ids: continue
            src = self.cell.all_synapses.synapse_sources[i]
            dst = self.cell.all_synapses.synapse_targets[i]
            w = self.cell.all_synapses.synapse_weights[i]
            
            src_name = inv_sensors[src] if src in inv_sensors else (f"Скрытый_{self.cell.neuron_ids.index(src)}" if src in self.cell.neuron_ids else f"ID_{src}")
            dst_name = inv_actions[dst] if dst in inv_actions else (f"Скрытый_{self.cell.neuron_ids.index(dst)}" if dst in self.cell.neuron_ids else f"ID_{dst}")
            print(f"  Синапс [{src_name} -> {dst_name}]: вес = {w:.2f}")


    def execute_cell_action(self, cell_instance, action: str) -> dict[str, float]:
        """
        Транслирует моторные команды `act` клетки в физическое движение на холсте
        и возвращает биологические последствия.
        """
        step = 4  # Скорость шага клетки в пикселях
        dx, dy = 0, 0
        consequences = {}

        # Обработка движения
        if action == "moveX+": dx = step
        elif action == "moveX-": dx = -step
        elif action == "moveY+": dy = step   # В графике Tkinter Y+ это вниз
        elif action == "moveY-": dy = -step  # Y- это вверх

        # Физическое перемещение с проверкой границ
        if dx != 0 or dy != 0:
            new_x = max(0, min(self.cell_x + dx, self.width - self.cell_size))
            new_y = max(0, min(self.cell_y + dy, self.height - self.cell_size))
            
            real_dx = new_x - self.cell_x
            real_dy = new_y - self.cell_y
            
            self.cell_x = new_x
            self.cell_y = new_y
            self.canvas.move(self.cell_graphic_id, real_dx, real_dy)

        # Обработка интеракций (eat / bite)
        # Проверяем, есть ли сейчас на карте зажатый мышкой объект
        active_objects = [obj for obj in self.entities.values()]
        
        if active_objects:
            obj = active_objects[0] # Берем тот объект, который сейчас держит пользователь
            
            # Считаем расстояние между центрами клетки и объекта мыши
            cx = self.cell_x + self.cell_size / 2
            cy = self.cell_y + self.cell_size / 2
            ox = obj["x"] + obj["size"] / 2
            oy = obj["y"] + obj["size"] / 2
            dist = math.hypot(cx - ox, cy - oy)
            
            # Условие контакта (пересечение кругов)
            is_inside = dist <= ((self.cell_size + obj["size"]) / 2) + 1.0


            if action == "eat" and obj["type"] == "food" and is_inside:
                consequences["energy"] = 0.35
                print("Клетка успешно съела еду!")
                
            elif action == "bite" and obj["type"] == "fire" and is_inside:
                consequences["health"] = -0.1
                print("Клетка обожглась при попытке укусить огонь!")

            if obj["type"] == "unknown_hazard" and is_inside:
                consequences["health"] = -0.1
                print("Клетка обожглась обо что-то синее и неизвестное! Punishment: -1.0")

        return consequences

    def generate_sensory_input(self) -> dict[str, float]:
        """
        Генерирует sensory_input на основе взаимного расположения клетки 
        и объекта, удерживаемого мышкой.
        """
        # Создаем пустой слепок сенсоров (все по нулям)
        sensory = {s: 0.0 for s in self.cell.sensors}
        
        # Если на экране нет зажатых мышкой объектов — сигналов нет
        active_objects = [obj for obj in self.entities.values()]
        if not active_objects:
            return sensory
            
        obj = active_objects[0]
        obj_type = obj["type"] # "food" или "fire"
        if obj_type == "food":
            prefix = "light_green"
        elif obj_type == "fire":
            prefix = "light_red"
        elif obj_type == "unknown_hazard":
            prefix = "light_blue" # Для синего объекта


        # Координаты центров
        cx = self.cell_x + self.cell_size / 2
        cy = self.cell_y + self.cell_size / 2
        ox = obj["x"] + obj["size"] / 2
        oy = obj["y"] + obj["size"] / 2

        dist = math.hypot(cx - ox, cy - oy)
        max_view_distance = 300.0 # Радиус чувствительности сенсоров

        # 1. Проверка на контакт (inside)
        if dist < (self.cell_size + obj["size"]) / 2:
            sensory[f"{prefix}_inside"] = 1.0
            return sensory # При контакте направленные лучи зануляем

        # 2. Расчет направленных осей (если объект в зоне видимости)
        if dist <= max_view_distance:
            # Интенсивность сигнала растет при приближении (от 0.0 до 1.0)
            intensity = 1.0 - (dist / max_view_distance)
            
            # Разница координат
            dx = ox - cx
            dy = oy - cy

            # Раскладываем вектор на сенсоры
            if dx > 0: sensory[f"{prefix}X+"] = intensity
            elif dx < 0: sensory[f"{prefix}X-"] = intensity

            if dy > 0: sensory[f"{prefix}Y+"] = intensity # Помним: Y+ в графике — это ВНИЗ
            elif dy < 0: sensory[f"{prefix}Y-"] = intensity # Y- — это ВВЕРХ

        return sensory

    def update_world(self):
        super().update_world()
        
        # 1. Формируем зрение для клетки на основе карты пикселей
        sensory = self.generate_sensory_input()
        
        # 2. Вызываем главный метод жизни клетки, передавая себя (SimulationWorld) как шлюз
        action, potential = self.cell.live(sensory, world_context=self)
        
        # Выводим телеметрию в консоль для отладки раз в 10 тактов, чтобы не спамить
        if self.tick_counter % 10 == 0 and any(sensory.values()):
            print(f"Такт {self.tick_counter} | Активные сенсоры: {[k for k,v in sensory.items() if v > 0]} | Действие: {action} | Энергия: {self.cell.state['energy']:.2f}")

        if self.tick_counter % 60 == 0:
            self.print_current_network_weights()