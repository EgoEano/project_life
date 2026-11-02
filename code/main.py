from cell.cell_gen1 import run as run_Gen1
from cell.cell_gen2_1 import run as run_Gen2_1
from cell.cell_gen2_D import run as run_Gen2_D
from cell.cell_gen2_2 import run as run_Gen2_2
from cell.cell_gen2_3 import run as run_Gen2_3
from cell.cell_gen3_0 import run as run_Gen3_0
from cell.cell_gen3_1 import run as run_Gen3_1
from cell.cell_gen3_2 import run as run_Gen3_2
from cell.cell_gen4_0 import run as run_Gen4_0
from cell.cell_gen4_1 import run as run_Gen4_1
from cell.cell_gen4_2 import run as run_Gen4_2
from cell.cell_gen4_3 import run as run_Gen4_3
from cell.cell_gen4_4 import run as run_Gen4_4
from world.my_simulation_v2 import MySimulationV2

# if __name__ == "__main__":
#     run_Gen4_4()


if __name__ == "__main__":
    sim = MySimulationV2()
    sim.run(fps=60)

