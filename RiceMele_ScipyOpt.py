import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm.auto import tqdm
import pandas as pd
from scipy import integrate
from scipy import constants
from scipy.optimize import minimize as minimize
from functools import partial
from IPython.display import display, clear_output
pi = np.pi
plt.rcParams['font.family']="arial unicode ms"
plt.rcParams['font.size']=14

#######################
#######constants#######
#######################
hbar = constants.hbar
c_light = constants.c
k_B = constants.k
mu_0 = constants.mu_0
solar_constant = 1366.1
e = constants.e
#######################
#######################

def H(k, hc):
    ham_k = np.zeros((2, 2), dtype=np.complex128)
    ham_k[0, 0] = 1
    ham_k[1, 1] = -1
    ham_k[0, 1] = hc["t"]*np.cos(k/2)-1j*hc["deltat"]*np.sin(k/2)
    ham_k[1, 0] = hc["t"]*np.cos(k/2)+1j*hc["deltat"]*np.sin(k/2)
    return ham_k
def pH(k, hc):
    t = hc["t"]
    deltat = hc["deltat"]
    ham_k = np.zeros((2, 2), dtype=np.complex128)
    ham_k[0, 0] = 0; ham_k[1, 1] = 0
    ham_k[0, 1] = -t/2*np.sin(k/2)-1j*deltat/2*np.cos(k/2)
    ham_k[1, 0] = -t/2*np.sin(k/2)+1j*deltat/2*np.cos(k/2)
    return ham_k
def ppH(k, hc):
    t = hc["t"]
    deltat = hc["deltat"]
    ham_k = np.zeros((2, 2), dtype=np.complex128)
    ham_k[0, 0] = 0; ham_k[1, 1] = 0
    ham_k[0, 1] = -t/4*np.cos(k/2)+1j*deltat/4*np.sin(k/2)
    ham_k[1, 0] = -t/4*np.cos(k/2)-1j*deltat/4*np.sin(k/2)
    return ham_k
def pppH(k, hc):
    t = hc["t"]
    deltat = hc["deltat"]
    ham_k = np.zeros((2, 2), dtype=np.complex128)
    ham_k[0, 0] = 0; ham_k[1, 1] = 0
    ham_k[0, 1] = t/8*np.sin(k/2)+1j*deltat/8*np.cos(k/2)
    ham_k[1, 0] = t/8*np.sin(k/2)-1j*deltat/8*np.cos(k/2)
    return ham_k

def fermiD(energy, beta, mu):
    return 1/(1+np.exp(beta*(energy-mu)))

def blackbody(omega):
    return omega**3/(np.exp(e*omega/(k_B*5500))-1)

def PVE_omega(omega, hc, beta, mu, gamma, N):
    PVE1, PVE2, PVE3, PVE4 = 0, 0, 0, 0
    for k in np.linspace(2*pi, 0, N, endpoint=False):
        ham = H(k, hc)
        val, vec = np.linalg.eig(ham)
        vec = vec.T
        for a in range(len(val)):
            PVE1 += fermiD(val[a], beta, mu)*np.vdot(vec[a], pppH(k, hc)@vec[a])
            for b in range(len(val)):
                PVE2 += (fermiD(val[a], beta, mu)-fermiD(val[b], beta, mu))*np.vdot(vec[a], pH(k, hc)@vec[b])*np.vdot(vec[b], ppH(k, hc)@vec[a])/(omega + 1j*gamma-(val[a]-val[b]))
                PVE2 += (fermiD(val[a], beta, mu)-fermiD(val[b], beta, mu))*np.vdot(vec[a], pH(k, hc)@vec[b])*np.vdot(vec[b], ppH(k, hc)@vec[a])/(-omega + 1j*gamma-(val[a]-val[b]))
                PVE3 += (fermiD(val[a], beta, mu)-fermiD(val[b], beta, mu))*np.vdot(vec[a], ppH(k, hc)@vec[b])*np.vdot(vec[b], pH(k, hc)@vec[a])/(1j*gamma-((val[a]-val[b])))
                for c in range(len(val)):
                    const = 2*np.vdot(vec[a], pH(k, hc)@vec[b])*np.vdot(vec[b], pH(k, hc)@vec[c])*np.vdot(vec[c], pH(k, hc)@vec[a])/(1j*gamma-(val[c]-val[a]))
                    PVE4 += const*(fermiD(val[a], beta, mu)-fermiD(val[b], beta, mu))/(omega+1j*gamma-(val[b]-val[a]))
                    PVE4 += const*(fermiD(val[c], beta, mu)-fermiD(val[b], beta, mu))/(omega+1j*gamma-(val[c]-val[b]))
                    PVE4 += const*(fermiD(val[a], beta, mu)-fermiD(val[b], beta, mu))/(-omega+1j*gamma-(val[b]-val[a]))
                    PVE4 += const*(fermiD(val[c], beta, mu)-fermiD(val[b],beta, mu))/(-omega+1j*gamma-(val[c]-val[b]))
    return (PVE1 + PVE2 + PVE3 + PVE4)*(-1/(N*omega**2))

sum_solar_radiation = integrate.quad(blackbody, 0.1, 10)
## objective function
def current(hc, beta, mu, gamma, N):
    Current = integrate.quad(lambda omega: PVE_omega(omega, hc, beta, mu, gamma, N).real*blackbody(omega)*2*mu_0*c_light*solar_constant/sum_solar_radiation[0], 0.1, 10)
    return Current

objective_function = lambda x, beta, mu, gamma, N: -current({"t": x[0], "deltat":x[1]}, beta, mu, gamma, N)[0]

def minimize_RiceMele(initial_val, beta, mu, N, gamma, method = "nelder-mead", maxiter = 400, tol=None):
    """
    input:
        initial_val: (float, float)
            initial parameters for the optimization.
            t: initial_val[0]
            deltat: initial_val[1]
        beta: float
            inverse temperature of the system
        mu: float
            chemical potential of the system
        N: int
            size of the system
            initiallyl set to N = 100
        gamma: float
            small value. set default by gamma = 2*pi*1e-13
        method: string
            choose from
                Nelder-Mead
                Powell
                CG
                BFGS
                Newton-CG
                L-BFGS-B
                TNC
                COBYLA
                COBYQA
                SLSQP
                dogleg
                trust-ncg
        maxiter: int
            maximum number of iteration
        tol: tolerance of the convergene

    output:
        result: scipy.optimize._optimize.OptimizeResult
            result of the optimization
        log: 2D array
            log[0]: parameter values for each iteration
            log[1]: objecrive functions for each iteration
    """
    global log
    log = [[], []] ## first array for storing the params, second array for storing objective function
    time_list = []
    objective_function_fixed_param = partial(objective_function, beta = beta, mu = mu, gamma = gamma, N=N)

    ## callback function for storing the optimization process
    def callback(x):
        time_list.append(time.time())
        log[0].append(x)
        log[1].append(objective_function_fixed_param(x))
        clear_output(wait=True)
        plt.plot(log[1], marker='o', color="blue")
        plt.xlabel('Iteration')
        plt.ylabel('Objective Function Value')
        plt.title('Objective Function Value During Optimization')
        plt.grid(True)
        display(plt.gcf())
        if len(time_list) > 1:
            print(f"iteration {len(time_list)} finished. Expected time: {(maxiter-len(time_list))*(time_list[-1]-time_list[-2])}")
        print("log:", log)
        print("time:", time_list)

    result = minimize(objective_function_fixed_param, initial_val, method = method, options={"disp":True, "maxiter":maxiter}, callback = callback, tol = tol)
    print("Iteration:", len(time_list))

    clear_output(wait=True)

    return result, log