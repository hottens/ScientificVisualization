import numpy as np
import sys
import math
from scipy.ndimage import gaussian_filter


# Simulation class
class Simulation:

    ### Initialize the simulation and its parameters
    def __init__(self, DIM):
        self.DIM = DIM
        self.dt = 0.4
        self.visc = 0.01
        self.field = np.zeros((3, DIM, DIM))  # vx, vy, rho
        self.field0 = np.zeros((3, DIM, DIM))
        self.field0c = np.zeros((2, int(DIM * (DIM / 2 + 1))), dtype=np.complex_)  # vx0, vy0, rho0

        self.forces = np.zeros((2, DIM, DIM))  # fx, fy

        self.divfield = np.zeros((DIM, DIM))
        self.divforces = np.zeros((DIM,DIM))

    ### Perform one iteration step with all needed operations
    def do_one_simulation_step(self, frozen):
        if not frozen:
            self.set_forces()
            self.solve()
            self.diffuse_matter()
            self.calc_divergence()

            self.values['force']['min'] = np.floor(np.amin(abs(self.forces[0, :, :] * self.forces[1, :, :])))
            self.values['force']['max'] = np.ceil(np.amax(abs(self.forces[0, :, :] * self.forces[1, :, :])))

            self.values['velo']['min'] = np.floor(np.amin(abs(self.field[0, :, :] * self.field[1, :, :])))
            self.values['velo']['max'] = np.ceil(np.amax(abs(self.field[0, :, :] * self.field[1, :, :])))

            self.values['rho']['min'] = np.floor(np.amin(self.field[-1, :, :]))
            self.values['rho']['max'] = np.ceil(np.amax(self.field[-1, :, :]))

            self.values['div_v']['min'] = np.floor(np.amin(self.divfield))
            self.values['div_v']['max'] = np.ceil(np.amax(self.divfield))

            self.values['div_f']['min'] = np.floor(np.amin(self.divforces))
            self.values['div_f']['max'] = np.ceil(np.amax(self.divforces))

    ### Define how forces evolve over time
    def set_forces(self):
        self.field0[-1, :, :] = self.field[-1, :, :] * 0.9
        self.forces[:2, :, :] = self.forces[:2, :, :] * 0.85
        self.field0[:2, :, :] = self.forces[:2, :, :]

    ### Define how the fluid / smoke velocities evolve over time
    def solve(self):
        DIM = self.DIM
        dt = self.dt
        visc = self.visc
        self.field[:2, :, :] += dt * self.field0[:2, :, :]
        self.field0[:2, :, :] = self.field[:2, :, :]
        U = np.zeros(2)
        V = np.zeros(2)

        for i in range(0, DIM):
            for j in range(0, DIM):
                x = (0.5 + i) / DIM
                y = (0.5 + j) / DIM
                x0 = DIM * (x - dt * self.field0[0, i, j]) - 0.5
                y0 = DIM * (y - dt * self.field0[1, i, j]) - 0.5
                i0 = clamp(x0)
                s = x0 - i0
                i0 = int((DIM + (i0 % DIM)) % DIM)
                i1 = int((i0 + 1) % DIM)

                j0 = clamp(y0)
                t = y0 - j0
                j0 = int((DIM + (j0 % DIM)) % DIM)
                j1 = int((j0 + 1) % DIM)
                s = 1 - s
                t = 1 - t
                self.field[:2, i, j] = (1 - s) * ((1 - t) * self.field0[:2, i0, j0] + t * self.field0[:2, i0, j1]) + s * (
                        (1 - t) * self.field0[:2, i1, j0] + t * self.field0[:2, i1, j1])
        for i in range(0, DIM):
            for j in range(0, DIM):
                self.field0[:2, i, j] = self.field[:2, i, j]

        self.field0cx = FFT(1, self.field0[0, :, :])
        self.field0cy = FFT(1, self.field0[1, :, :])
        
        for i in range(0, int(DIM / 2 + 1), 1):
            y = 0.5 * i
            for j in range(0, DIM):
                x = j if j <= DIM / 2 else j - DIM
                r = x * x + y * y
                if r == 0.0:
                    continue
                f = np.exp(-r * dt * visc)
                U[0] = self.field0cx[j, i].real
                V[0] = self.field0cy[j, i].real
                U[1] = self.field0cx[j, i].imag
                V[1] = self.field0cy[j, i].imag

                self.field0cx[j, i] = complex(f * ((1 - x * x / r) * U[0] - x * y / r * V[0]),
                                         f * ((1 - x * x / r) * U[1] - x * y / r * V[1]))
                self.field0cy[j, i] = complex((f * (-y * x / r * U[0] + (1 - y * y / r) * V[0])),
                                         (f * (-y * x / r * U[1] + (1 - y * y / r) * V[1])))

        self.field0[0, :, :] = FFT(-1, self.field0cx)
        self.field0[1, :, :] = FFT(-1, self.field0cy)

        f = 1.0
        for i in range(0, DIM):
            for j in range(0, DIM):
                self.field[:2, i, j] = f * self.field0[:2, i, j]

    ### Define how the fluid / smoke moves through the field
    def diffuse_matter(self):
        DIM = self.DIM
        dt = self.dt
        for i in range(0, DIM):
            for j in range(0, DIM):
                x = (0.5 + i) / DIM
                y = (0.5 + j) / DIM
                x0 = DIM * (x - dt * self.field[0, i, j]) - 0.5
                y0 = DIM * (y - dt * self.field[1, i, j]) - 0.5
                i0 = clamp(x0)
                s = x0 - i0
                i0 = int((DIM + (i0 % DIM)) % DIM)
                i1 = int((i0 + 1) % DIM)

                j0 = clamp(y0)
                t = y0 - j0

                j0 = int((DIM + (j0 % DIM)) % DIM)
                j1 = int((j0 + 1) % DIM)

                self.field[-1, i, j] = (1 - s) * ((1 - t) * self.field0[-1, i0, j0] + t * self.field0[-1, i0, j1]) + s * (
                        (1 - t) * self.field0[-1, i1, j0] + t * self.field0[-1, i1, j1])

    ### Calculate the divergences of velocities and forces
    def calc_divergence(self):
        DIM = self.DIM
        for i in range(0,DIM):
            for j in range(0,DIM):
                self.divfield[i,j] = -( self.field[0,i-1,j]-self.field[0,i,j] + self.field[0,i,j] - self.field[0,(i+1)%DIM,j] + self.field[1,i,j-1]-self.field[1,i,j] +  self.field[1,i,j] - self.field[1,i,(j+1)%DIM])
                self.divforces[i,j] = -(self.forces[0,i-1,j]-self.forces[0,i,j] + self.forces[0,(i+1)%DIM,j]-self.forces[0,i,j] + self.forces[1,i,j-1]-self.forces[1,i,j] + self.forces[1,i,(j+1)%DIM]- self.forces[1,i,j])


### Clamp
def clamp(x):
    return int(x) if x >= 0.0 else int(x - 1)


### Fast Fourier Transform
def FFT(direction, v):
    return np.fft.rfft2(v) if direction == 1 else np.fft.irfft2(v)


