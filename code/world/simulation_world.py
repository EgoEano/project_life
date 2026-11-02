import tkinter as tk
import time
import uuid

class SimulationWorld:
    def __init__(self, width: int = 1000, height: int = 1000):
        self.width = width
        self.height = height

        self.root = tk.Tk()
        self.root.title("SNN & R-STDP Simulation World")
        self.root.resizable(False, False) # Запрещаем менять размер окна

        # 2. Создаем черный холст (Canvas) для отрисовки
        self.canvas = tk.Canvas(self.root, width=width, height=height, bg="black", highlightthickness=0)
        self.canvas.pack()
        
        # Переменная для подсчета тактов мира
        self.tick_counter = 0

        self.entities = {}


    def spawn_object(self, obj_type: str, x: int, y: int, size: int = 6, color: str = "white") -> uuid.UUID:
        """
        Регистрирует объект в мире и сразу рисует его на холсте.
        obj_type: например, "cell", "food", "fire"
        """
        # Ограничиваем координаты границами экрана
        x = max(0, min(x, self.width - size))
        y = max(0, min(y, self.height - size))
        
        # Создаем визуальный объект на холсте Tkinter (овал/круг)
        # Он возвращает целочисленный ID, по которому Tkinter умеет им управлять
        graphic_id = self.canvas.create_oval(x, y, x + size, y + size, fill=color, outline="")
        
        # Генерируем уникальный ID для вашей логики
        obj_id = uuid.uuid4()
        
        # Сохраняем в базу данных мира
        self.entities[obj_id] = {
            "type": obj_type,
            "x": x,
            "y": y,
            "size": size,
            "color": color,
            "graphic_id": graphic_id
        }
        
        return obj_id


    def move_object(self, obj_id: uuid.UUID, dx: int, dy: int):
        """
        Безопасно перемещает объект на вектор (dx, dy) с учетом границ.
        """
        if obj_id not in self.entities:
            return
            
        obj = self.entities[obj_id]
        
        # Считаем новые координаты
        new_x = max(0, min(obj["x"] + dx, self.width - obj["size"]))
        new_y = max(0, min(obj["y"] + dy, self.height - obj["size"]))
        
        # Считаем реальную дельту с учетом утыкания в края
        real_dx = new_x - obj["x"]
        real_dy = new_y - obj["y"]
        
        # Обновляем координаты в нашей памяти
        obj["x"] = new_x
        obj["y"] = new_y
        
        # Приказываем Tkinter сдвинуть графический объект на дельту
        # Это работает моментально, так как Tkinter не перерисовывает весь экран
        self.canvas.move(obj["graphic_id"], real_dx, real_dy)


    def remove_object(self, obj_id: uuid.UUID):
        """Удаляет объект из памяти и с экрана (например, если еду съели)"""
        if obj_id in self.entities:
            # Удаляем с холста
            self.canvas.delete(self.entities[obj_id]["graphic_id"])
            # Удаляем из памяти
            del self.entities[obj_id]


    def update_world(self):
        """
        Главный игровой цикл (каждый вызов — один такт симуляции)
        """
        # Увеличиваем счетчик времени
        self.tick_counter += 1
        
        # --- ТУТ БУДЕТ ВАША КЛЕТКА И ОБЪЕКТЫ ---
        # Пока холст просто остается черным
        # --------------------------------------

        # Обновляем окно графического интерфейса
        self.root.update_idletasks()
        self.root.update()

    def run(self, fps: int = 60):
        """Запуск бесконечного цикла симуляции с ограничением FPS"""
        frame_time = 1.0 / fps
        
        # Флаг для корректного закрытия окна пользователем
        running = True
        def on_closing():
            nonlocal running
            running = False
            self.root.destroy()
            
        self.root.protocol("WM_DELETE_WINDOW", on_closing)

        # Главный цикл
        while running:
            start_time = time.time()
            
            self.update_world()
            
            sleep_time = frame_time - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)