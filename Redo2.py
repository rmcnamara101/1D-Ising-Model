import math as math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import pandas as pd
import time
import random


font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }

class VortexSim():
    """
    Vortex simulation
    """
    def __init__(self, positions: list, circulations: list, num_steps: int) -> None:
        """
        Initialises Simulation and required constants
        """
        self.h_m = 0.001
        self.annihilation_distance = 0.1
        self.epsilon = 0.01
        self.max_step = 0.001
        self.original_max_step = 0.001
        self.adjusted_max_step = False
        self.boundary_threshold = 0.01
        self.vortex_boundary = False
        self.positions = np.array(positions, dtype=complex)
        self.circulations = np.array([self.h_m * x for x in circulations])
        self.num_steps = num_steps
        self.circ_history = []
    
    def dzkdt(self, k, z) -> float:
        """
        Calculates the velocity dzk/dt 
        """
        real_interaction = 0
        other_image_vortices = 0

        zk = self.positions[k]
        gamma_k = self.circulations[k]
        
        for j in range(len(self.positions)):
            if j != k:
                zj = self.positions[j]
                gamma_j = self.circulations[j]

                real_interaction += -gamma_k * gamma_j / ((zk - zj))
                other_image_vortices += gamma_k * gamma_j * zj.conjugate() / ( zk * zj.conjugate() - 1)
        
        own_image_interaction = -gamma_k**2 * zk.conjugate() / (1 -  zk * zk.conjugate())

        derivative =  np.pi*(real_interaction + other_image_vortices+ own_image_interaction)

        return derivative

    def equations_of_motion(self, t, z) -> list:
        """
        Defines the equations of motion 
        """
        num_vortices = len(self.positions)
        velocities = np.zeros(2*num_vortices, dtype=float)  

        for k in range(num_vortices):
           
            dH_dzk = self.dzkdt(k,z)
            
            z_dot_k = -2j * dH_dzk / self.circulations[k]

            
            velocities[2*k] = -z_dot_k.real
            velocities[2*k + 1] = z_dot_k.imag
            

        return velocities

    def vortex_annihilation(self) -> None:
        """
        Method to handle the annihilation of the vortices
        """
        vortices_to_remove = set()

        for i in range(len(self.positions)):
            for j in range(i + 1, len(self.positions)):
                separation = abs(self.positions[i] - self.positions[j])
                if separation <= self.annihilation_distance and self.circulations[i] != self.circulations[j]:
                    vortices_to_remove.add(i)
                    vortices_to_remove.add(j)

        self.positions = [v for i, v in enumerate(self.positions) if i not in vortices_to_remove]
        self.circulations = [c for i, c in enumerate(self.circulations) if i not in vortices_to_remove]

    def vortex_ode(self, t, y) -> list:
        """
        Shapes data from equations_of_motion() function to fit into the differential equation solver
        """
        self.positions = y.view(complex)  
        velocities = self.equations_of_motion(t, y)
        self.circ_history.append((t, self.circulations))
        
        print(f"Curent Time: {t}")
        return velocities
      
    def boundary_event(self, t, y) -> float:
        """
        Defines the conditions for an event when a vortex leaves the boundary so that the simulation can be stopped
        """

        positions = y.reshape((-1, 2))
        distances_to_boundary = 1 - np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
        min_distance_to_boundary = np.min(distances_to_boundary)

        positions = y.reshape((-1,2))
        for i in range(len(positions)):
            r = np.sqrt(positions[i][0]**2 + positions[i][1]**2)
            if r > 1:
                self.positions = positions
    
                return 0

        return 1

    boundary_event.terminal = True

    def annihilation_event(self, t, y) -> float:
        """
        Defines the conditions for an annihilation event so that integration can stop
        """
        positions = y.reshape((-1, 2))
    
        min_distance = np.inf
        
        for i in range(len(self.circulations)):
            for j in range(i+1, len(self.circulations)):
                if self.circulations[i] * self.circulations[j] < 0: 
                    distance = np.linalg.norm(positions[i] - positions[j])
                    if distance < min_distance:
                        min_distance = distance
                        
        if min_distance <= self.annihilation_distance:
            return 0
        else:
            return 1

    annihilation_event.terminal = True  # Stops integration when event is triggered

    def vortex_near_boundary(self, y):
        """
        Checks if any vortex is within a specified distance from the boundary.
        """
        num_vortices = len(y) // 2
        positions = y.reshape((-1,2))
        distances = []
        for i in range(num_vortices):
            x, y = positions[i][0], positions[i][1]
            distance_to_origin = np.sqrt(x**2 + y**2)
            distance_to_boundary = abs(distance_to_origin - 1)
            distances.append(distance_to_boundary)
            
    
        if min(distances) < self.boundary_threshold:
            self.vortex_boundary = True
        else:
            self.vortex_boundary = False
            
    def boundary_proximity_event(self, t, y):
        """
        Triggers an event if any vortex is within a specified distance from the boundary.
        """
        self.vortex_near_boundary(y)
        trip = self.vortex_boundary
        if trip == True and self.adjusted_max_step == False:
            return 0
        elif trip == False and self.adjusted_max_step == True:
            return 0 
        else:
            return 1  

    boundary_proximity_event.terminal = False

    def simulate(self, t_span: list, max_step: float) -> list:
        """
        Runs the vortex simulation over the specified time range (unless a vortex leaves the boundary)
        """
        t_start, t_end = t_span
        t_current = t_start
        all_solutions = []
        desired_num_points = 100
        all_t_evals = []

        def complex_to_real(complex_positions):
                num_vortices = len(complex_positions)
                real_positions = np.zeros(2 * num_vortices, dtype=float)

                for k in range(num_vortices):
                    real_positions[2*k] = complex_positions[k].real
                    real_positions[2*k + 1] = complex_positions[k].imag

                return real_positions
        
        initial_conditions = np.array(complex_to_real(self.positions)).flatten()
        
        
        while t_current < t_end:
            initial_conditions = np.array(complex_to_real(self.positions)).flatten()
            current_time_step = (t_end - t_current) / desired_num_points
            next_t_eval = np.arange(t_current, t_end, current_time_step)
           
            next_t_eval = next_t_eval[next_t_eval < t_end]

            sol = solve_ivp(
                    self.vortex_ode,
                    (t_current, t_end),
                    initial_conditions,
                    t_eval=next_t_eval,
                    events=[self.annihilation_event, self.boundary_event, self.boundary_proximity_event], 
                    max_step=self.max_step,
                    rtol=1e9,
                    atol=1e6,
                    method='RK45'
                )
    
            all_solutions.append(sol)
            all_t_evals.extend(next_t_eval.tolist())
    
        
            if sol.t_events[0].size > 0:
                # An annihilation event occurred
                annihilation_time = sol.t_events[0][0] + self.max_step
                t_current = annihilation_time
                print('annihilation!')
                self.vortex_annihilation()

            elif sol.t_events[1].size > 0:
                # A vortex left the boundary 
                print('A vortex left the boundary at time:', sol.t_events[1][0])
                leave_time = sol.t_events[1][0]
                t_current = leave_time - self.max_step
                self.reposition_vortex()
                #break

            #adaptive time step method
            elif sol.t_events[2].size > 0 and self.vortex_boundary == True:
                # Trigger adaptive step size
                self.max_step = 0.001
                self.adjusted_max_step = True
                trigger_time = sol.t_events[2][0] + self.original_max_step
                t_current = trigger_time

            elif sol.t_events[2].size > 0 and self.vortex_boundary == False:
                # Remove adaptive step size
                self.max_step = self.original_max_step
                self.adjusted_max_step = False
                trigger_time = sol.t_events[2][0]
                t_current = trigger_time
                
            else:
                t_current = t_end # simulation ended naturally

            def extract_circulations_at_sol_times(sol_t, circ_history):
                circulations_at_sol_times = []
                circ_history_times = [time for time, _ in circ_history]
                
                for t in sol_t:
                    idx = min(range(len(circ_history_times)), key=lambda i: abs(circ_history_times[i]-t))
                    circulations_at_sol_times.append(circ_history[idx][1])
                    
                return circulations_at_sol_times
            
            circ_history = extract_circulations_at_sol_times(sol.t, self.circ_history)
            

        time_points = np.array(all_t_evals)
        
        final_circ = []
        for i in range(len(self.circulations)):
            final_circ.append(self.circulations[i])

        return all_solutions, time_points, circ_history, t_current
    
    def create_animation(self, all_positions: list, circulations: list):
        """
        Method to create animation of the evolution of the system
        """
        fig, ax = plt.subplots()
        
        def update(frame):
            ax.clear()
            positions = all_positions[frame]
            frame_circulations = circulations[frame]

            vortex_colors = ['blue' if c > 0 else 'red' for c in frame_circulations]
            
            ax.scatter(positions[:, 0], positions[:, 1], c=vortex_colors, marker=',')
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.add_artist(plt.Circle((0, 0), 1, color='black', fill=False))
            ax.set_title(f'Iteration: {frame}')
            
        ani = FuncAnimation(fig, update, frames=len(all_positions), repeat=True)
        plt.show()

    def calculate_energy(self, vortex_positions: list, circulations: list) -> float:
        """
        Method to calculate the energy of particular microstate using the hamiltonian
        """
        energy = 0 
        
        pos = [x + y*1j for x, y in vortex_positions]

        for i in range(len(pos)):
            zi = pos[i]
            energy += (1/4*np.pi)*circulations[i]**2*np.log(1-zi*zi.conjugate())
            for j in range(len(pos)):
                if i < j:
                    zj = pos[j]
                    energy += -(1/4*np.pi)*circulations[i]*circulations[j]*np.log((zi - zj)*(zi-zj).conjugate())
                    energy += (1/4*np.pi)*circulations[i]*circulations[j]*np.log((1-zi*zj.conjugate())*(1-zj*zi.conjugate()))

        return energy.real / len(vortex_positions)
    
    def calculate_angular_momentum(self, vortex_positions: list, circulations: list) -> float:
        """
        Method to calculate the net angular momentum of the system
        """
        L = 0

        positions = [x + y*1j for x, y in vortex_positions]

        for i in range(len(positions)):
            zi = positions[i]
            L += math.pi*(1-abs(zi)**2)*circulations[i]
        
        return L / len(positions)


#Data Analysis Functions
def process_solutions(solutions: list) -> list:
    """
    Structures the solutions into all positions at each time step of the simulation
    """
    all_positions = []
    for sol in solutions:
        for i in range(sol.y.shape[1]):
            positions_at_t = sol.y[:, i].reshape(-1, 2)
            all_positions.append(positions_at_t)
    return all_positions

def calculate_energy(vortex_positions: list, circulations: list) -> float:
        """
        Method to calculate the energy of particular microstate using the hamiltonian
        """
        energy = 0 
        pos = [x + y*1j for x, y in vortex_positions]

        for i in range(len(pos)):
            zi = pos[i]
            energy += (1/4*np.pi)*circulations[i]**2*np.log(1-zi*zi.conjugate())
            for j in range(len(pos)):
                if i < j:
                    zj = pos[j]
                    energy -= (1/4*np.pi)*circulations[i]*circulations[j]*np.log((zi - zj)*(zi.conjugate()-zj.conjugate()))
                    energy += (1/4*np.pi)*circulations[i]*circulations[j]*np.log((1-zi*zj.conjugate())*(1-zj*zi.conjugate()))

        return abs(energy) / len(circulations)

def calculate_angular_momentum(vortex_positions: list, circulations: list) -> float:
        """
        Method to calculate the net angular momentum of the system
        """
        
        positions = vortex_positions#[x + y*1j for x, y in vortex_positions]
        
        L = 0

        for i in range(len(vortex_positions)):
            zi = positions[i]
            L += math.pi*(1-zi*zi.conjugate())*circulations[i]
        
        return L.real / len(vortex_positions)

def get_entropy_temp(energy: float, angular_momentum: float, entropy: str) -> float:
        """
        Method to calculate the entropy of a particular microstate
        """
        df = pd.read_excel(entropy)
        energies = df.iloc[:, 0].tolist()
        angular_momentums = df.columns.tolist()[1:]
        energy_difference = []
        l_dif = []

        for i in range(0,99):
            e = abs(energies[i] - energy)
            energy_difference.append(e)
            l = abs(angular_momentums[i] - angular_momentum)
            l_dif.append(l)

        e_min = min(energy_difference)
        l_min = min(l_dif)
        e_index = energy_difference.index(e_min)
        l_index = l_dif.index(l_min)

        entropy = df.iloc[e_index , l_index+1]

        return entropy

def plot_intitial_state(positions: list, circulations: list, j: float):
    fig, ax = plt.subplots()
    pos = positions
    circ = circulations

    for i in range(len(pos)):
        if circ[i] == 1:
            colour = 'blue'
        else:
            colour = 'red'

        ax.scatter(pos[i][0], pos[i][1], color=colour)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.add_artist(plt.Circle((0, 0), 1, color='black', fill=False))
    fig.savefig(f'Projects/PHYS3900 Superfluid Vortices/Experiment Initial State/Initial state {j}')

def plot_final_state(positions: list, circulations: list, j: float):
    fig, ax = plt.subplots()
    pos = positions
    circ = circulations

    for i in range(len(pos)):
        if circ[i] == 1 or circ[i] == 0.001:
            colour = 'blue'
        else:
            colour = 'red'

        ax.scatter(pos[i][0], pos[i][1], color=colour)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.add_artist(plt.Circle((0, 0), 1, color='black', fill=False))
    fig.savefig(f'Projects/PHYS3900 Superfluid Vortices/Experiment Final State/Final state {j}')

def dipole_moment(positions: list, circulations: list):
    d = 0
    def convert_to_complex(positions):
        return [x + y*1j for x, y in positions]
    pos = convert_to_complex(positions)
    for i in range(len(pos)):
        d += circulations[i]*pos[i]

    return d*d.conjugate()

def save_observable(t, observable, name, i):
    fig, ax = plt.subplots()
    ax.grid(True)

    if name == 'Entropy':
        ax.plot(t, observable)
        ax.set_ylabel('Entropy [k]')
        ax.set_xlabel('Time')
        ax.set_title(f'Entropy over simulation {i}')
    elif name == 'Energy':
        ax.plot(t, observable)
        ax.set_ylabel('Energy [Jh/mN]')
        ax.set_xlabel('Time')
        ax.set_title(f'Energy over simulation {i}')
    elif name == 'Dipole Moment':
        ax.plot(t, observable)
        ax.set_ylabel('Dipole Moment [hd/m]')
        ax.set_xlabel('Time')
        ax.set_title(f'Dipole Moment over simulation {i}')
    elif name == 'Temperature':
        ax.plot(t, observable)
        ax.set_ylabel('Temperature [kNm/Jh]')
        ax.set_xlabel('Time')
        ax.set_title(f'Temperature over simulation {i}')
    elif name == 'Angular Momentum':
        ax.plot(t, observable)
        ax.set_ylabel('Angular Momentum [h/md^2]')
        ax.set_xlabel('Time')
        ax.set_title(f'Angular Momentum over simulation {i}')

    fig.savefig(f'Projects/PHYS3900 Superfluid Vortices/Tester')


def main() -> None:
    """
    Experimental function
    """
    def convert_to_complex(positions):
        return [x + y*1j for x, y in positions]

    def random_position(radius):
        r = random.random() * radius 
        theta = random.random() * 2 * math.pi  
        x = r * math.cos(theta)  
        y = r * math.sin(theta) 
        return [x, y]

    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


    def is_too_close(position, positions, min_distance):
        return any(distance(position, existing_position) < min_distance for existing_position in positions)


    for j in range(0,1):

        num_vortices = 2
        min_distance = 0.3 
        pos1 = []
        while len(pos1) < num_vortices:
            new_position = random_position(0.8)
            if not is_too_close(new_position, pos1, min_distance):
                pos1.append(new_position)
        circ = [1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1]
        start_time = time.time()

        pos1 = [[0.2,0],[-0.2,0]]
        circ = [1,-1]
        pos = convert_to_complex(pos1)
        energy_hist = []
        num_steps = 150

        plot_intitial_state(pos1, circ, j)

        sim = VortexSim(pos, circ, num_steps)
        sol = sim.simulate((0,num_steps), 0.01)
        circ_history = sol[2]
    

        all_positions = process_solutions(sol[0])
        mpositionst = []
        entropy_hist = []
        l_hist = []
        time_steps = np.linspace(0,num_steps,len(all_positions))
        temp_hist =[]
        dipole_hist = []
    

        for i in range(len(all_positions)):
    
            if i >= len(circ_history):
                circ_history.append(circ_history[-1])
            
            num_vortices = len(all_positions[i])
            num_circulations = len(circ_history[i])
            

            while num_vortices > num_circulations:
                circ_history[i].extend([0.001, -0.001])
                num_circulations = len(circ_history[i])


        for i in range(len(all_positions)):
            energy = calculate_energy(all_positions[i], circ_history[i])
            energy_hist.append(energy)
            l = calculate_angular_momentum(all_positions[i], circ_history[i])
            l_hist.append(l)
            t = get_entropy_temp(energy, l, 'updated_entropy.xlsx')
            s = get_entropy_temp(energy, l, 'updated_temp.xlsx')
            entropy_hist.append(s)
            temp_hist.append(t)
            dipole = dipole_moment(all_positions[i], circ_history[i])
            dipole_hist.append(dipole)

        initial_energy = energy_hist[0]
        final_energy = energy_hist[-1]
        initial_l = l_hist[0]
        final_l = l_hist[-1]
        initial_entropy = get_entropy_temp(initial_energy, 0, 'updated_entropy.xlsx')
        final_entropy = get_entropy_temp(final_energy, 0, 'updated_entropy.xlsx')
        initial_temp = get_entropy_temp(initial_energy, 0, 'updated_temp.xlsx')
        final_temp = get_entropy_temp(final_energy, 0, 'updated_temp.xlsx')
        initital_dipole = dipole_hist[0]
        final_dipole = dipole_hist[-1]

        print("Process finished --- %s seconds ---" % (time.time() - start_time))

        txt = f'Initial energy: {initial_energy} \n Final Energy per Vortex: {final_energy}\n Final Number of Vortices: {len(all_positions[-1])} \nInitial Dipole: {initital_dipole}\nFinal Dipole: {final_dipole}'

        filename = f"Projects/PHYS3900 Superfluid Vortices/Experimental Data/experiment round 3.5 0.55r{j}_results_{sol[3]}_time_steps.txt"


        with open(filename, 'w') as file:
            file.write(txt)

        save_observable(time_steps, energy_hist, 'Energy', j)
        save_observable(time_steps, entropy_hist, 'Entropy', j)
        save_observable(time_steps, dipole_hist, 'Dipole Moment', j)
        save_observable(time_steps, l_hist, 'Angular Momentum', j)
        save_observable(time_steps, temp_hist, 'Temperature', j)

    print(energy_hist)

    sim.create_animation(all_positions, circ_history)
    plt.plot(time_steps, energy_hist)
    plt.show()
    plt.plot(time_steps, dipole_hist)
    plt.show()
    
    
    pass

if __name__ == "__main__":
    main()
