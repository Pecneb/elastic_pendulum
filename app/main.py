#!/usr/bin/env python
# coding: utf-8

# Rezges-Lenges szimulacio

import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import argparse
import scipy.spatial.distance as distance

# User interface for easier use
'''
age: rezges-lenges_PeterBenceX89O8X.py [-h] [-D SPRING_CONSTANT] [-L SPRING_LENGTH] [-m MASS_OF_BODY] [-x X] [-y Y] [-z Z] [--chaoticD CHAOTICD] [--chaoticL CHAOTICL] [--chaoticM CHAOTICM]

Elastic pendulum simulation. Spring constant, spring length, the mass of the body and the starting coordinatescan be set with the flags above. Otherwise DEFAULT values are used. example usage: python3 rezges-lenges_PeterBenceX89O8X.py --chaoticL -D 15.0 -m 0.5 -x 0.05 -z -0.03

optional arguments:
    -h, --help            show this help message and exit
    -D SPRING_CONSTANT, --spring_constant SPRING_CONSTANT
    default value 50.0
    -L SPRING_LENGTH, --spring_length SPRING_LENGTH
    default value 1.2
    -m MASS_OF_BODY, --mass_of_body MASS_OF_BODY
    default value 2.0
    -x X                  starting coordinate X, default value 0.0
    -y Y                  starting coordinate Y, default value 0.0
    -z Z                  starting coordinate Z, default value -1.0
    --chaoticD CHAOTICD   flag to enable chaotic movement, and calc D from m and L
    --chaoticL CHAOTICL   flag to enable chaotic movement, and calc L from m and D
    --chaoticM CHAOTICM   flag to enable chaotic movement, and calc m from D and L
'''
parser = argparse.ArgumentParser(description='Elastic pendulum simulation. Spring constant, spring length, the mass of the body and the starting coordinatescan be set with the flags above. Otherwise DEFAULT values are used. example usage: python3 rezges-lenges_PeterBenceX89O8X.py --chaoticL -D 15.0 -m 0.5 -x 0.05 -z -0.03')
parser.add_argument('-D', '--spring_constant',  default=50.0, type=np.float64, help='default value 50.0')
parser.add_argument('-L', '--spring_length', default=1.2, type=np.float64, help='default value 1.2')
parser.add_argument('-m', '--mass_of_body', default=2.0, type=np.float64, help='default value 2.0')
parser.add_argument('-x', default=0.0, type=np.float64, help='starting coordinate X, default value 0.0')
parser.add_argument('-y', default=0.0, type=np.float64, help='starting coordinate Y, default value 0.0')
parser.add_argument('-z', default=-1.0, type=np.float64, help='starting coordinate Z, default value -1.0')
parser.add_argument('--chaoticD', default=False, action='store_true', help='flag to enable chaotic movement, and calc D from m and L')
parser.add_argument('--chaoticL', default=False, action='store_true', help='flag to enable chaotic movement, and calc L from m and D')
parser.add_argument('--chaoticM', default=False, action='store_true', help='flag to enable chaotic movement, and calc m from D and L')
parser.add_argument('--dimensions', default=2, choices=[2,3], type=int, help='Set the animation dimensions.')
parser.add_argument('--drag', default=False, action='store_true', help='Use flag --drag, when want to calculate with drag force.')
parser.add_argument('--Tratio', default=1, type=np.float64, help='Period T time ratio.')

args = parser.parse_args()

# Force of Gravity (vector format) 
G = np.array([0,0,-9.81], dtype=np.float64) # Can vary, based on where are you on earth.

# Linear body constant (at low velocity!)
C_lin = 1.0

# Chaotic movement can be achieved when Tpendulum and Tspring ratio is irrational (PI multiplier is hardcoded here)
if args.chaoticD:
    # Default Length L
    L = args.spring_length
    # The mass of the body at the end of the elastic pendulum
    m = args.mass_of_body
    D = (1 / args.Tratio**2) * ((m * (G**2).sum()**0.5) / L)
elif args.chaoticL:
    # Default Spring constant
    D = args.spring_constant
    # The mass of the body at the end of the elastic pendulum
    m = args.mass_of_body
    L = (1 / args.Tratio**2) * ((m * (G**2).sum()**0.5) / D)
elif args.chaoticM:
    # Default Length L
    L = args.spring_length
    # Default Spring constant D
    D = args.spring_constant
    m = (args.Tratio**2) * ((L * D) / (G**2).sum()**0.5)
else:
    # Default Length L
    L = args.spring_length
    # The mass of the body at the end of the elastic pendulum
    m = args.mass_of_body
    # Default Spring constant D
    D = args.spring_constant

FIX = np.array([0.0, 0.0, L]) # rugos / inga rogzitesi pontja

# Egyszeru mozgasegyenlet-megoldo
def lepes(xn, vn, m, F, dt):
    a = F(xn, vn, m) / m
    v_new = vn + a*dt
    x_new = xn + v_new*dt
    return (x_new, v_new)

# 3D rugo
def F_rugo(r, v, m):
    return (-D*r)

# 3D inga
def F_inga(r, v, m):
    # gravitacios ero:
    F_erinto = m * G * np.sin(r/L)
    return (F_erinto)


# 3D rugo linearis kozegellenallassal
def F_rugo_kozeg_lin(r, v, m):
    F = F_rugo(r, v, m) - C_lin*v
    return F

# 3D rugos-inga
def F_rugos_inga(r, v, m):
    F = F_rugo(r, v, m) + (m * G * np.sin(r/np.linalg.norm(FIX-r)))
    return F

# 3D rugos-inga kozegellenallassal
def F_rugos_inga_kozeg(r, v, m):
    F = F_rugo_kozeg_lin(r, v, m) + (m * G * np.sin(r/np.linalg.norm(FIX-r)))
    return F

def app():
    x0 = np.array([args.x, args.y, args.z], dtype=np.float64)
    v0 = np.array([0,0,0], dtype=np.float64)
    dt = 0.01

    if args.drag:
        F_fuggveny = F_rugos_inga_kozeg # erofuggveny kivalasztasa
    elif not args.drag:
        F_fuggveny = F_rugos_inga

    x = x0; v = v0
    t = 0.0; t_max = 100.0

    t_list = []
    x_list = []
    v_list = []

    # szamitas
    while(t<=t_max+1e-6):
        t_list.append(t)
        v_list.append(v)
        x_list.append(x)
        x, v = lepes(x, v, m, F_fuggveny, dt)
        t += dt
        
    # tömbbé alakítjuk
    x_arr=np.asarray(x_list)
    v_arr=np.asarray(v_list)
    t_arr=np.asarray(t_list)

    # most kirajzoljuk
    """
    fig=plt.figure(figsize=(18,12))  
    ax1=fig.add_subplot(211) # két rész-grafikon
    ax2=fig.add_subplot(212)
    ax1.plot(t_list, x_arr[:,0], color="red") # az elsőbe az rx(t)
    ax1.plot(t_list, x_arr[:,1], color="blue")
    ax1.plot(t_list, x_arr[:,2], color="green")
    ax2.plot(t_list, v_arr[:,0], color="red")# a másodikba a vx(t)
    ax2.plot(t_list, v_arr[:,1], color="blue")
    ax2.plot(t_list, v_arr[:,2], color="green")
    ax1.grid()
    ax2.grid()
    ax1.set_xlabel("t")
    ax1.set_ylabel("x(t)")
    ax2.set_xlabel("t")
    ax2.set_ylabel("v_x(t)")
    """

    # 3D plot
    """
    fig2 = plt.figure(figsize=(15,15))
    ax1 = fig2.add_subplot(111, projection='3d')
    ax1.plot(x_arr[:,0], x_arr[:,1], x_arr[:,2])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    """
    
    if args.dimensions == 2:
        # 2D animation plot
        history_len = x_arr.shape[0]
        fig3 = plt.figure(figsize=(15,15))
        axf3 = fig3.add_subplot()
        axf3.grid()

        line, = axf3.plot([], [], '.-', lw=1, ms=2)
        body, = axf3.plot([], [], 'ro')
        axf3.set(xlim=(x_arr[:,:].min(), x_arr[:,:].max()), xlabel='X')
        axf3.set(ylim=(x_arr[:,:].min(), x_arr[:,:].max()), ylabel='Z')
        time_template = 'time = %.1fs'
        time_text = axf3.text(0.05, 0.9, s='', transform=axf3.transAxes)

        def animate(i, x, z):
            line.set_data(x[:i], z[:i])
            body.set_data(np.array([x[i]]), np.array([z[i]]))
            time_text.set_text(time_template % (i*dt))
            return line, body, time_text

        ani = animation.FuncAnimation(
                fig3, animate, fargs=(x_arr[:,0], x_arr[:,2]),
        interval=dt, blit=True)
    else:
        # 2D animation plot
        history_len = x_arr.shape[0]
        fig3 = plt.figure(figsize=(15,15))
        axf3 = fig3.add_subplot(projection='3d')
        axf3.grid()

        line, = axf3.plot([], [], [], '.-', lw=1, ms=2)
        body, = axf3.plot([], [], [], 'ro')
        axf3.set(xlim3d=(x_arr[:,:].min(), x_arr[:,:].max()), xlabel='X')
        axf3.set(ylim3d=(x_arr[:,:].min(), x_arr[:,:].max()), ylabel='Y')
        axf3.set(zlim3d=(x_arr[:,:].min(), x_arr[:,:].max()), zlabel='Z')
        time_template = 'time = %.1fs'
        time_text = axf3.text(0.05, 0.05, 0.9, s='', transform=axf3.transAxes)

        def animate(i, x, y, z):
            line.set_data(x[:i], y[:i])
            line.set_3d_properties(z[:i])
            body.set_data(np.array([x[i]]), np.array([y[i]]))
            body.set_3d_properties(np.array([z[i]]))
            time_text.set_text(time_template % (i*dt))
            return line, body, time_text

        ani = animation.FuncAnimation(
                fig3, animate, fargs=(x_arr[:,0], x_arr[:,1], x_arr[:,2]),
        interval=dt, blit=True)

    V_Fig = plt.figure(figsize=(15,15))
    V_ax = V_Fig.add_subplot()
    V_ax.grid()
    V_ax.plot(v_arr[:,2], x_arr[:,2])
    plt.show()

def main():
    app() 

if __name__ == "__main__":
    main()
