import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def no_drag():
    v = int(input(" Enter initial Velocity\n")) # Initial velocity
    g = 9.81  #gravity
    angle = int(input("Specify an angle\n")) #Angle input
    theta = angle / 180.0*np.pi #convert to radians
    plt.figure()
    tmax = ((2 * v) * np.sin(theta)) / g #Calculate maximum time
    print(f"Flight time: {tmax}")
    timemat = tmax*np.linspace(0,1,100)[:,None] # defines the step size
    x = ((v * timemat) * np.cos(theta)) # calculate X component
    y = ((v * timemat) * np.sin(theta)) - ((0.5 * g) * (timemat ** 2)) # calculate Y component
    plt.plot(x,y) #plot X,Y
    plt.xlabel('Range [m]')
    plt.ylabel('Height [m]')
    plt.grid()
    plt.show()
    calc()


def with_drag():
    c = float(input("Enter drag coefficent"))
    r = float(input("Enter radious of the object [m]")) #radious of the object
    A = np.pi * r ** 2 #area of the object
    m = float(input("Enter mass of the object [kg]")) #object mass
    rho_air = 1.28 #Air density
    g = 9.81 #acceleration g
    k_const = 0.5 * c * rho_air * A #Const used in the equations
    #get initial values
    theta = np.radians(65)
    v0 = int(input(" Enter initial Velocity\n")) # Initial velocity

    angle = int(input("Specify an angle\n")) #Angle input
    theta = angle / 180.0*np.pi #convert to radians

    def deriv(t, u): #Equations of motion
        x, xdot, y, ydot = u
        speed = np.hypot(xdot, ydot)
        xdotdot = -k_const / m * speed * xdot
        ydotdot = -k_const / m * speed * ydot - g
        return xdot, xdotdot, ydot, ydotdot


    u0 = 0, v0 * np.cos(theta), 0., v0 * np.sin(theta) #initial values
    t0, tf = 0, 50

    def iszero(t, u): #check if we're back at surface level
        return u[2]

    iszero.terminal = True
    iszero.direction = -1

    def max_height(t, u):
        #Peak reached?
        return u[3]
    soln = solve_ivp(deriv, (t0, tf), u0, dense_output=True,events=(iszero, max_height))
    print(soln)
    t = np.linspace(0, soln.t_events[0][0], 100)
    sol = soln.sol(t)
    x, y = sol[0], sol[2]
    print(f'xmax = {x[-1]} m')
    print(f'ymax = {max(y)} m')
    plt.plot(x, y)
    plt.xlabel('Range [m]')
    plt.ylabel('Height [m]')
    plt.grid()
    plt.show()
    calc()

def calc():
    include_drag = input("Calculate with (Y) or without drag (N)?\n")
    if include_drag =="N":
        no_drag()
    else:
        with_drag()

calc()