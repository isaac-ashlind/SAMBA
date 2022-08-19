"""
      ______   ______  __       __ _______   ______
     /      \ /      \|  \     /  \       \ /      \
    |  ▓▓▓▓▓▓\  ▓▓▓▓▓▓\ ▓▓\   /  ▓▓ ▓▓▓▓▓▓▓\  ▓▓▓▓▓▓\
    | ▓▓___\▓▓ ▓▓__| ▓▓ ▓▓▓\ /  ▓▓▓ ▓▓__/ ▓▓ ▓▓__| ▓▓    Study
     \▓▓    \| ▓▓    ▓▓ ▓▓▓▓\  ▓▓▓▓ ▓▓    ▓▓ ▓▓    ▓▓    of
     _\▓▓▓▓▓▓\ ▓▓▓▓▓▓▓▓ ▓▓\▓▓ ▓▓ ▓▓ ▓▓▓▓▓▓▓\ ▓▓▓▓▓▓▓▓    Axial
    |  \__| ▓▓ ▓▓  | ▓▓ ▓▓ \▓▓▓| ▓▓ ▓▓__/ ▓▓ ▓▓  | ▓▓    Magnetic
     \▓▓    ▓▓ ▓▓  | ▓▓ ▓▓  \▓ | ▓▓ ▓▓    ▓▓ ▓▓  | ▓▓    Bottle
      \▓▓▓▓▓▓ \▓▓   \▓▓\▓▓      \▓▓\▓▓▓▓▓▓▓ \▓▓   \▓▓    Asymmetry

Isaac AshLind
PPPL Summer 2022
"""

import os
import warnings
import numpy as np
import pandas as pd
import astropy.units as u
from astropy import constants
from plasmapy.particles import Particle
from plasmapy.formulary import Coulomb_logarithm, thermal_speed
from pqdm.processes import pqdm
from datetime import datetime


def main():
    """
    choose to follow menu prompts or hard coded program
    """
    menu()
    #auto()


def menu():
    """
    menu prompt version of program
    """
    print("SAMBA (Study of Axial Magnetic Bottle Asymmetry)")
    print("************************************************")
    params_flag = params_prompt()
    if params_flag:
        params = load_csv()
    else:
        params_message()
        params = get_params()
    coll_flag = coll_prompt()
    particles = get_particles(coll_flag)
    grid_flag = grid_prompt()
    if grid_flag:
        grid = load_csv()
    else:
        grid = build_grid()
    t, N = duration_number(coll_flag)
    iterable = build_iterable(params, particles,
                              grid, t, coll_flag)
    simulation(N, iterable, coll_flag)
    print("PROGRAM END")


def auto():
    """
    hard coded version of program
    """
    params = pd.read_csv("params3.csv", index_col=0)
    grid = pd.read_csv("grid.csv", index_col=0)
    particles = (Particle("e-"), Particle("p+"))
    t, N = (4, 100)
    collisions = True
    iterable = build_iterable(params, particles,
                              grid, t, collisions)
    simulation(N, iterable, collisions)
    print("PROGRAM END")


# SUPPORTING FUNCTIONS IN ALPHABETICAL ORDER

def boris(X, V, B, dt, q, m, _0, _1, _2, _3, _4):
    """
    collisionless particle pusher
    """
    U = q / m * B * dt / 2
    W = 2 * U / (1 + np.dot(U, U))
    V_prime = V + np.cross(V, U)
    V_new = V + np.cross(V_prime, W)
    X_new = X + V_new * dt / 2
    return X_new, V_new


def B_r(r, z, B_0, b, d):
    """
    B_x and B_y are equivalent to B_r, just replace r
    """
    temp1 = 2 * B_0 * d ** 2 * r * z * np.log(b)
    temp2 = z ** 4 - d ** 4
    temp3 = z ** 4 + d ** 4
    temp4 = 2 * (z * d) ** 2
    return temp1 * temp2 * b ** (temp4 / temp3) / temp3 ** 2


def B_z(z, B_0, b, d):
    return B_0 * b ** (2 * (z * d) ** 2 / (z ** 4 + d ** 4))


# TODO: let lists be different lengths
def build_grid():
    text="""
    instructions:
    1) build parameter grid with lists of initial values
    2) include commas between values and nothing else
    3) all lists must be the same length
    * angles in degrees and particles start with z = y = 0
    """
    print(text)
    columns = ["radii", "speed", "pitch", "phase"]
    success = False
    while success == False:
        try:
            print("radii (relative to the system radius):")
            radii = np.array(input().split(","), dtype=float)
            print("speed (relative to thermal velocity):")
            speed = np.array(input().split(","), dtype=float)
            print("pitch angle (measure from positive z):")
            pitch = np.array(input().split(","), dtype=float)
            print("phase angle (measure CCW in xy-plane):")
            phase = np.array(input().split(","), dtype=float)
            grid = np.array([radii, speed, pitch, phase]).T
            grid_df = pd.DataFrame(grid, columns=columns)
            success = True
        except:
            print("\ntry again")
    save_csv(grid_df)
    return grid_df


# TODO: replace thermal_speed to phase out plasmapy formulary dependence
def build_iterable(params_series, particles, grid_df, t, coll_flag):
    """
    pqdm needs input in the form of a single
    iterable for parallel processing
    """
    params = params_series.to_numpy().flatten()
    n = params[5] / u.m ** 3
    T = params[6] * u.keV
    R = params[7]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v_therm = thermal_speed(T, particles[0]).value # m / s
    kappa = collision_constant(n, T, particles)
    nu = n.value * kappa / v_therm ** 3 # Hz
    t_total = t / nu # collision time at origin
    grid = grid_df.to_numpy().T
    radii, speed, pitch, phase = np.meshgrid(*grid)
    meshgrid = np.array([radii.ravel() * R,
                         speed.ravel() * v_therm,
                         pitch.ravel() * np.pi / 180,
                         phase.ravel() * np.pi / 180],
                         dtype=object)
    cat1 = np.full((meshgrid.shape[1], len(params)), params).T
    cat2 = np.array([np.full(meshgrid.shape[1], particles[0]),
                     np.full(meshgrid.shape[1], kappa),
                     np.full(meshgrid.shape[1], v_therm),
                     np.full(meshgrid.shape[1], t_total),
                     np.full(meshgrid.shape[1], coll_flag)])
    return np.concatenate((meshgrid, cat1, cat2), axis=0)
    # iterable must be transposed before using


def coll_prompt():
    print("\ninclude pitch angle scattering collisions?")
    print("\t(Y) --> must be electrons scattering off of ions")
    print("\t(N) --> may choose any particle")
    answer = None
    while answer not in ("Y", "y", "N", "n"):
        answer = input("Y/N: ")
    if answer in ("Y", "y"):
        return True
    else:
        return False


def collision_constant(n, T, particles):
    """
    calculate collision constant
    as a function of density and temp
    """
    e_0 = constants.eps0.value # F / m
    m_a = particles[0].mass.value # kg
    q_a = particles[0].charge.value # C
    q_b = particles[1].charge.value # C
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ln_lambda = Coulomb_logarithm(T, n, particles)
    temp1 = (q_a * q_b) ** 2 * ln_lambda
    temp2 = 4 * np.pi * (e_0 * m_a) ** 2
    return temp1 / temp2


def coulomb(X, V, B, dt, q, m, kappa, v_therm, n, T, rng):
    """
    collisional particle pusher
    """
    v_norm = np.linalg.norm(V)
    nu = n * kappa / v_norm ** 3
    D = nu * v_therm ** 3 / v_norm
    dW = np.sqrt(dt) * rng.standard_normal(size=3)
    temp1 = np.sqrt(D) * np.cross(V, dW) / (2 * v_norm ** 2)
    temp2 = q / m * B * dt / 2
    M = temp1 - temp2
    M_hat = np.array([[   0 ,-M[2], M[1]],
                      [ M[2],   0 ,-M[0]],
                      [-M[1], M[0],   0]])
    temp3 = np.identity(3) - M_hat
    temp4 = np.identity(3) + M_hat
    caley = np.matmul(np.linalg.inv(temp3), temp4)
    V_new = np.matmul(caley, V)
    X_new = X + V_new * dt / 2
    return X_new, V_new


def density(n_0, r, R):
    """
    plasma density as a function of radius
    """
    n = n_0 * (1 - (r / R) ** 2)
    return np.where(r < R, n, 0)


def duration_number(coll_flag):
    """
    get duration and number of simulations
    """
    t = None
    N = None
    while type(t) != float:
        try:
            print("\nduration (in collision times): ")
            t = float(input())
        except:
            print("please provide a number (float or int)")
    if coll_flag:
        while type(N) != int:
            try:
                print("\nnumber of simulations: ")
                N = int(input())
            except:
                print("please provide an integer")
    else:
        N = 1
    return t, N


def get_B(x, y, z, B_0, b_neg, b_pos, d_neg, d_pos):
    """
    get B field as a function of position
    """
    neg = np.array([z, B_0, b_neg, d_neg])
    pos = np.array([z, B_0, b_pos, d_pos])
    args = np.where(z < 0, neg, pos)
    return np.array([B_r(x, *args), B_r(y, *args), B_z(*args)])


def get_params():
    """
    get B field and plasma parameters
    """
    params = {"B_0": None, "b_neg": None, "b_pos": None,
              "d_neg": None, "d_pos": None,
              "n_0": None, "T": None, "R": None}
    for key in params.keys():
        while type(params[key]) != float:
            try:
                params[key] = float(input(f"{key}: "))
            except:
                print("please provide a number (float or int)")
    params_series = pd.Series(params)
    save_csv(params_series)
    return params_series


def get_particles(coll_flag):
    print("\nwhether or not collisions are included,")
    print("time is normalized by collision frequency")
    if not coll_flag:
        part_a = None
        while part_a == None:
            try:
                part_a = Particle(input("\nchoose particle_a: "))
            except:
                print("not a valid particle")
    else:
        print("\nparticle_a = electron")
        part_a = Particle("electron")
    part_b = None
    while part_b == None:
        try:
            part_b = Particle(input("choose particle_b: "))
        except:
            print("not a valid particle")
    return part_a, part_b


def grid_prompt():
    print("\nload initial parameter grid from file?")
    answer = None
    while answer not in ("Y", "y", "N", "n"):
        answer = input("Y/N: ")
    if answer in ("Y", "y"):
        return True
    else:
         return False


def load_csv():
    while True:
        try:
            filename = input("filename: ") + ".csv"
            data = pd.read_csv(filename, index_col=0)
            break
        except:
            print("try again")
    return data


def params_message():
    text = """
    Please provide the following parameters:
    B_0 = basal magnetic field value at z = 0 (T)
    b_neg = multiple of B_0 for bottle end at z < 0
    b_pos = multiple of B_0 for bottle end at z > 0
    d_neg = distance from B_0 (z = 0) to end z < 0 (m)
    d_pos = distance from B_0 (z = 0) to end z > 0 (m)
    n_0 = plasma density at origin (1 / m^3)
    T = plasma temperature in terms of energy (keV)
    R = radius of mirror machine system (m)
    """
    print(text)


def params_prompt():
    print("\nload machine/plasma parameters from file?")
    answer = None
    while answer not in ("Y", "y", "N", "n"):
        answer = input("Y/N: ")
    if answer in ("Y", "y"):
        return True
    else:
        return False


def save_csv(data):
    print("\nsave data to file?")
    answer = None
    while answer not in ("Y", "y", "N", "n"):
        answer = input("Y/N: ")
    if answer in ("Y", "y"):
        data.to_csv(input("filename: ") + ".csv")


def simulation(N, iterable, coll_flag):
    """
    batch function for Monte Carlo simulation
    """
    batch_size = len(iterable.T)
    columns = ["x", "y", "z", "r", "v", "theta", "phi", "t"]
    results_df = pd.DataFrame(index=range(N * batch_size),
                              columns=columns)
    for n in range(N):
        print(f"\n{n + 1}/{N}")
        # seed for repeatable results
        seed = np.full(iterable.shape[1], n)
        args = np.concatenate((iterable, [seed]), axis=0).T
        batch = pqdm(args, trajectory, n_jobs = os.cpu_count())
        for i, row in enumerate(batch):
            results_df.iloc[n * batch_size + i] = row
    # TODO: improve date format for results filename
    results_df.to_csv(f"results_{datetime.today()}.csv")
    print("results saved to file!")


def stitch(x, d_neg, d_pos):
    """
    particles exiting bottle enter new bottle
    """
    L = d_neg + d_pos
    return (x + d_neg) % L - d_neg


def trajectory(args):
    """
    simulate single particle trajectory
    """
    num = 4 # number of evaluation points per radian
    r, v, theta, phi = args[:4]
    B_0, b_neg, b_pos, d_neg, d_pos = args[4:9]
    n_0, T, R = args[9:12]
    p_a, kappa = args[12:14]
    v_therm, t_total = args[14:16]
    coll_flag, seed = args[16:]
    q_a = p_a.charge.value # C
    m_a = p_a.mass.value # kg
    if coll_flag:
        # create random generator 
        rng = np.random.default_rng(seed)
    else:
        rng = None
    # initialize velocity
    v_perp = v * np.sin(theta)
    v_x = v_perp * np.cos(phi)
    v_y = v_perp * np.sin(phi)
    v_z = v * np.cos(theta) # v_parallel for z = 0
    V = np.array([v_x, v_y, v_z])
    # initialize position 
    X = np.array([r, 0, 0])
    # determine particle pusher
    if coll_flag:
        pusher = coulomb
    else:
        pusher = boris
    t_elapsed = 0
    B = get_B(*X, *args[4:9])
    while t_elapsed < t_total:
        # find local gyrofrequency
        B_norm = np.linalg.norm(B)
        w_gyro = np.abs(q_a) * B_norm / m_a
        dt = 1 / w_gyro / num
        X_ahead = X + V * dt / 2
        # use bottle length modulus to get periodic B-field
        stitch_z = stitch(X_ahead[2], d_neg, d_pos)
        B = get_B(X_ahead[0], X_ahead[1], stitch_z, *args[4:9])
        n = density(n_0, r, R, B_norm, B_0)
        X, V = pusher(X_ahead, V, B, dt, q_a, m_a,
                     kappa, v_therm, n, T, rng)
        t_elapsed += dt
    return np.array([X[0], X[1], X[2], r, v,
                     theta, phi, t_elapsed])


if __name__ == "__main__":
    main()
