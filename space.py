import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime, timedelta
from mpl_toolkits.mplot3d import Axes3D  
from skyfield.api import load, Loader
MASS = {
    'sun': 1.98847e30, 'mercury': 3.3011e23, 'venus': 4.8675e24,
    'earth': 5.97237e24, 'mars': 6.4171e23, 'jupiter': 1.8982e27,
    'saturn': 5.6834e26, 'uranus': 8.6810e25, 'neptune': 1.02413e26,
    'pluto': 1.303e22
}

SF_NAMES = {
    'sun': 'sun', 'mercury': 'mercury barycenter', 'venus': 'venus barycenter',
    'earth': 'earth barycenter', 'mars': 'mars barycenter',
    'jupiter': 'jupiter barycenter', 'saturn': 'saturn barycenter',
    'uranus': 'uranus barycenter', 'neptune': 'neptune barycenter',
    'pluto': 'pluto barycenter'
}

AU_M = 1.495978707e11  


def compute_positions(eph, bodies, dt, frame="barycentric"):
    ts = load.timescale()
    t = ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    sun = eph['sun']

    pos = []
    for b in bodies:
        body = eph[SF_NAMES[b]]
        vec = np.array(body.at(t).position.au)  
        if frame == "heliocentric":
            vec -= np.array(sun.at(t).position.au)
        pos.append(vec * AU_M)
    return np.array(pos) 


def compute_barycenter(positions, bodies):
    masses = np.array([MASS[b] for b in bodies])
    weighted = (positions.T * masses).T
    return weighted.sum(axis=0) / masses.sum()


def animate_simulation(eph, bodies, start_dt, frame):
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Solar System 3D Animation")

    scat = ax.scatter([], [], [], s=40, c="blue")
    text_labels = [ax.text(0, 0, 0, b.capitalize()) for b in bodies]
    bary_point, = ax.plot([], [], [], "rx", markersize=10, label="Barycenter")

    frames = 200
    times = [start_dt + timedelta(days=i * 5) for i in range(frames)]

    def init():
        ax.set_xlim(-5 * AU_M, 5 * AU_M)
        ax.set_ylim(-5 * AU_M, 5 * AU_M)
        ax.set_zlim(-5 * AU_M, 5 * AU_M)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        return scat, bary_point, *text_labels

    def update(i):
        dt = times[i]
        positions = compute_positions(eph, bodies, dt, frame)
        bary = compute_barycenter(positions, bodies)

        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        scat._offsets3d = (x, y, z)  
        bary_point.set_data([bary[0]], [bary[1]])
        bary_point.set_3d_properties([bary[2]])

        for j, txt in enumerate(text_labels):
            txt.set_position((x[j], y[j]))
            txt.set_3d_properties(z[j])
        ax.set_title(f"Solar System Animation (3D)\n{dt.strftime('%Y-%m-%d')}")
        return scat, bary_point, *text_labels

    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False, interval=100)
    plt.legend()
    plt.show()


def run_animation():
    try:
        date_str = date_entry.get()
        frame = frame_var.get()
        selected_bodies = [b for b, var in body_vars.items() if var.get() == 1]
        if not selected_bodies:
            messagebox.showerror("Error", "Select at least one body.")
            return
        dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")

        loader = Loader("~/.skyfield-data")
        eph = loader("de440s.bsp")

        animate_simulation(eph, selected_bodies, dt, frame)

    except Exception as e:
        messagebox.showerror("Error", str(e))

root = tk.Tk()
root.title("Solar System 3D Animator")

tk.Label(root, text="Enter Date/Time (YYYY-MM-DD HH:MM:SS):").grid(row=0, column=0, sticky="w")
date_entry = tk.Entry(root, width=25)
date_entry.insert(0, datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
date_entry.grid(row=0, column=1)

tk.Label(root, text="Frame:").grid(row=1, column=0, sticky="w")
frame_var = tk.StringVar(value="barycentric")
ttk.Combobox(root, textvariable=frame_var, values=["barycentric", "heliocentric"]).grid(row=1, column=1)

tk.Label(root, text="Select Bodies:").grid(row=2, column=0, sticky="w")
body_vars = {}
row = 3
for b in SF_NAMES.keys():
    var = tk.IntVar(value=1 if b in ["sun", "earth", "jupiter"] else 0)
    chk = tk.Checkbutton(root, text=b.capitalize(), variable=var)
    chk.grid(row=row, column=0, sticky="w")
    body_vars[b] = var
    row += 1

tk.Button(root, text="Run 3D Animation", command=run_animation).grid(row=row, column=0, columnspan=2, pady=10)

root.mainloop()