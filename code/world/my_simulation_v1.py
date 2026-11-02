import random

from world.simulation_world import SimulationWorld



class MySimulationV1(SimulationWorld):
    def __init__(self):
        super().__init__(1200, 600)
        
        # Создаем статичные объекты (еда - зеленая, огонь - красный)
        for _ in range(20):
            self.spawn_object("food", random.randint(50, 950), random.randint(50, 950), size=8, color="green")
        for _ in range(5):
            self.spawn_object("fire", random.randint(50, 950), random.randint(50, 950), size=15, color="red")
            
        # Создаем нашу клетку (синяя) и запоминаем её ID
        self.my_cell_id = self.spawn_object("cell", 500, 500, size=12, color="cyan")

    def update_world(self):
        # Вызываем базовое обновление (счетчик тактов)
        super().update_world()
        
        # Симулируем хаотичное движение клетки каждый такт
        # В реальной задаче здесь SNN вашей клетки решит, куда сделать шаг
        dx = random.choice([-3, 0, 3])
        dy = random.choice([-3, 0, 3])
        
        self.move_object(self.my_cell_id, dx, dy)
