import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from IPython.display import display, clear_output

# =========================
# Physical Parameters
# =========================
m1, m2 = 1.0, 0.5          # Masses
L1, L2 = 1.0, 0.75         # Lengths
k1, k2 = 40, 40            # Spring constants
g = 9.81                   # Gravity
alp = 0.1                  # Damping coefficient
psi4, F = 0, 0             # Extra parameters (unused in this version)

# =========================
# Differential Equations
# =========================
# State vector z = [theta1, theta2, theta1_dot, theta2_dot, r1, r2, r1_dot, r2_dot]

def coupled_pendulum(t, z):
    z0, z1, z2, z3, z4, z5, z6, z7 = z
    Z = z0 - z2

    # Mass matrix A
    A = np.diag([1, 1, m1 * z4, m2 * z5, 1, 1, m1, -m2])
    A[3, 2] = m2 * z4 * np.cos(Z)
    A[3, 6] = m2 * np.sin(Z)
    A[7, 2] = m2 * z4 * np.sin(Z)
    A[7, 6] = -m2 * np.cos(Z)

    # Forcing vector B
    B = np.array([
        z2,
        z3,
        (-2 * m1 * z6 - alp * z4) * z2 - k2 * (z5 - L2) * np.sin(Z) - g * m1 * np.sin(z0),
        -m2 * (2 * np.cos(Z * z6 * z2) + 2 * z7 * z3 + g * np.sin(z1))
        - alp * z4 * np.cos(Z * z2) - alp * z5 * z3 - alp * np.sin(Z * z6),
        z6,
        z7,
        m1 * z4 * z2 ** 2 - k1 * (z4 - L1) + k2 * (z5 - L2) * np.cos(Z) + g * m1 * np.cos(z0) - alp * z6,
        -m2 * (z4 * np.cos(Z * z2 ** 2) + z5 * z3 ** 2 + 2 * np.sin(Z * z6 * z2) - g * np.cos(z1))
        + k2 * (z5 - L2) - alp * (z4 * np.sin(Z * z2) + np.cos(Z * z6) + z7)
    ])

    return np.linalg.solve(A, B)

# =========================
# Simulation Function
# =========================
def run_simulation(z0, periods=5, n_points=200):
    tau = 2 * np.pi * (np.sqrt(L1 / g) + np.sqrt(L2 / g))  # Approx. period
    t_span = (0, periods * tau)
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    return solve_ivp(coupled_pendulum, t_span, z0, method='Radau', t_eval=t_eval)

# =========================
# Plotting Functions
# =========================
def plot_angles_and_radii(sol):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Angles
    ax1.plot(sol.t, sol.y[0], label='theta1')
    ax1.plot(sol.t, sol.y[1], label='theta2')
    ax1.set_ylabel('Angle (rad)')
    ax1.legend()

    # Radii
    ax2.plot(sol.t, sol.y[4], label='r1')
    ax2.plot(sol.t, sol.y[5], label='r2')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Radius (m)')
    ax2.legend()

    plt.show()

def animate_trajectory(sol):
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    line1, = ax.plot([], [], 'o-', label='theta1')
    line2, = ax.plot([], [], 'o-', label='theta2')
    ax.legend()

    def update(frame):
        x1 = L1 * np.sin(sol.y[0, frame])
        y1 = -L1 * np.cos(sol.y[0, frame])
        x2 = x1 + L2 * np.sin(sol.y[1, frame])
        y2 = y1 - L2 * np.cos(sol.y[1, frame])

        line1.set_data([0, x1, x2], [0, y1, y2])
        line2.set_data([x1, x2], [y1, y2])
        display(fig)
        clear_output(wait=True)
        return line1, line2

    for frame in range(len(sol.t)):
        update(frame)
        plt.pause(0.05)

    plt.show()

# =========================
# Tension Analysis (Hooke's Law)
# =========================
def analyze_tensions(sol, periods=5):
    T1 = k1 * (sol.y[4] - L1)
    T2 = k2 * (sol.y[5] - L2)
    T = T1 + T2

    t = np.linspace(0, periods * (2 * np.pi * (np.sqrt(L1 / g) + np.sqrt(L2 / g))), len(T))
    plt.plot(t, T, label='Tension')
    plt.xlabel('Time (s)')
    plt.ylabel('Tension (N)')
    plt.legend()
    plt.show()

    # Max & min tensions per period
    n = periods
    dT = len(T) // periods
    tens_max, tens_min = [], []
    for i in range(n):
        segment = np.abs(T[dT * i:dT * (i + 1)])
        tens_max.append(np.max(segment))
        tens_min.append(np.min(segment))

    print(f"Max tensions per period: {tens_max}")
    print(f"Min tensions per period: {tens_min}")

# =========================
# Main Execution
# =========================
if __name__ == "__main__":
    # Initial conditions: low energy case
    z0_case1 = [np.pi / 6, np.pi / 4, 0, 0, L1, L2, 0, 0]
    sol1 = run_simulation(z0_case1)
    plot_angles_and_radii(sol1)
    animate_trajectory(sol1)
    analyze_tensions(sol1)

    # Second case: different initial conditions (chaotic behavior)
    z0_case2 = [np.pi / 2, -np.pi / 4, 0.5, 0.7, L1, L2, 0.6, 0.2]
    sol2 = run_simulation(z0_case2)
    plot_angles_and_radii(sol2)
    animate_trajectory(sol2)
    analyze_tensions(sol2)
