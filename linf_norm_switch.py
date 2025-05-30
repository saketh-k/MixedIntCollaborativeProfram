import sys

import numpy as np
import pulp as lp
from pulp.mps_lp import writeLP


class TerrainPlanner:
    def __init__(self, terrain_str: str, total_length: float = 120) -> None:
        self.terrain = np.array(list(terrain_str))
        self.Ns = len(self.terrain)
        self.ds = total_length / self.Ns

        # Robot Efficiencies in (J/kg/m)
        self.robot1_eff = np.where(self.terrain == "L", 1.0, 3.0)
        self.robot2_eff = np.where(self.terrain == "L", 3.0, 1.0)

        # Robot Masses (kg)
        self.robot1_mass = 1
        self.robot2_mass = 1

    def optimize(self, switch_cost):
        prob = lp.LpProblem("Robot_Path_Optimization", lp.LpMinimize)

        # Decision variables:
        z1 = [lp.LpVariable(f"z1_{i}", cat="Binary") for i in range(self.Ns)]
        z2 = [lp.LpVariable(f"z2_{i}", cat="Binary") for i in range(self.Ns)]
        z3 = [lp.LpVariable(f"z3_{i}", cat="Binary") for i in range(self.Ns)]
        switches = [ lp.LpVariable(f"same_z1_{i}", cat="Binary") for i in range(self.Ns - 1) ]
        t_max = [lp.LpVariable(f"t_max{i}",lowBound=0) for i in range(self.Ns)]
        t1 = [lp.LpVariable(f"t1{i}",lowBound = 0) for i in range(self.Ns)]
        t2 = [lp.LpVariable(f"t2{i}",lowBound = 0) for i in range(self.Ns)]

        # Enforce switching cost
        for i in range(self.Ns - 1):
            prob += switches[i] >= z1[i] - z1[i + 1]
            prob += switches[i] >= z1[i + 1] - z1[i]
            prob += switches[i] >= z2[i] - z2[i + 1]
            prob += switches[i] >= z2[i + 1] - z2[i]
            prob += switches[i] >= z3[i] - z3[i + 1]
            prob += switches[i] >= z3[i + 1] - z3[i]

        # Objectives
        for i in range(self.Ns):
            prob += t1[i] == self.robot1_eff*self.ds* (self.robot1_mass * z1[i] + (self.robot1_mass + self.robot2_mass) * z2[i])
            prob += t_max[i] == self.robot1_eff*self.ds* (self.robot2_mass * z1[i] + (self.robot1_mass + self.robot2_mass) * z3[i])
            # prob += t_max[i] >=t1[i]
            # prob += t_max[i] >=t2[i]
            prob += z1[i] + z2[i] + z3[i] == 1

        prob += lp.lpSum(t_max) >= lp.lpSum(t1)
        prob += lp.lpSum(t_max) >= lp.lpSum(t2)
        prob +=  lp.lpSum(t_max) + switch_cost * lp.lpSum(switches)
        prob.solve()

        assignment = np.array(
            [
                1 * z1[i].varValue + 2 * z2[i].varValue + 3 * z3[i].varValue # pyright: ignore
                for i in range(self.Ns)
            ]
        )
        total_energy = np.array([lp.lpSum(t1).value(),lp.lpSum(t2).value()])
        writeLP(prob, "lp_file")

        return assignment, total_energy

    def visualize(self, assignment, ascii_shell: bool = True):
        if not ascii_shell:
            COLORS = {
                "reset": "",
                "3": "",
                "2": "",
                "1": "",
                "0": "",
                "L": "",
                "W": "",
            }
        else:
            COLORS = {
                "L": "\033[93m",  # Yellow for Land
                "W": "\033[94m",  # Blue for
                "1": "\033[92m",  # Green for Mode
                "2": "\033[91m",  # Red for Mode
                "3": "\033[95m",  # Purple for Mode
                "reset": "\033[0m",
            }
        for i in range(self.Ns):
            terrain_color = COLORS["L"] if self.terrain[i] == "L" else COLORS["W"]
            mode_color = COLORS[str(int(assignment[i]))]
            print(
                f"{terrain_color}{self.terrain[i]}{mode_color}{int(assignment[i])}{COLORS['reset']}",
                end="  ",
            )
            if (i % 20) == 19:
                print()

if __name__ == "__main__":
    sw_cost = 5 if len(  sys.argv ) <= 2 else sys.argv[1]
    terrain = "LLLLLLLLLLLLLLL" * 8
    planner = TerrainPlanner(terrain, 1)
    assignment, energy = planner.optimize(float(sw_cost))

    print(f"Robot 1 Energy: {energy[0]:.2f} J")
    print(f"Robot 2 Energy: {energy[1]:.2f} J")

    planner.visualize(assignment, False)
