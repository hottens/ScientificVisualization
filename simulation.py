import numpy as np
import sys
import math
from scipy.ndimage import gaussian_filter


# Simulation
class Simulation:

    def __init__(self, DIM):
        self.values = {'rho': {'min': 0, 'max': 0},
                       'velo': {'min': 0, 'max': 0},
                       'force': {'min': 0, 'max': 0},
                       'div_v': {'min': 0, 'max': 0},
                       'div_f': {'min': 0, 'max': 0}}

        self.sinkholes = []
        self.DIM = DIM
        self.dt = 0.4
        self.visc = 0.01
        self.field = np.zeros((3, DIM, DIM))  # vx, vy, rho
        self.field0 = np.zeros((3, DIM, DIM))
        self.field0c = np.zeros((2, int(DIM * (DIM / 2 + 1))), dtype=np.complex_)  # vx0, vy0, rho0

        self.forces = np.zeros((2, DIM, DIM))  # fx, fy

        self.divfield = np.zeros((DIM, DIM))
        self.divforces = np.zeros((DIM,DIM))

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

    def set_forces(self):
        self.field0[-1, :, :] = self.field[-1, :, :] * 0.9
        self.forces[:2, :, :] = self.forces[:2, :, :] * 0.85
        self.field0[:2, :, :] = self.forces[:2, :, :]

        m = np.matrix([0.013, 0.108, 0.242, 0.0, -0.242, -0.108, -0.013])
        fx = np.dot(m.T, [[1, 2, 3, 4, 3, 2, 1]])
        fy = np.dot(np.matrix([1, 2, 3, 4, 3, 2, 1]).T, m)
        for [x, y, size] in self.sinkholes:
            self.field0[-1, x:x+2, y:y+2] = self.field[-1, x:x+2, y:y+2] * 0.8
            self.forces[0,x-1:x+3,y-1:y+3] = self.forces[0,x-1:x+3,y-1:y+3] + matrixy
            self.forces[1,x-1:x+3,y-1:y+3] = self.forces[1,x-1:x+3,y-1:y+3] + matrixx
            # for t in range(1,size):
            #     f = t
            #     self.forces[0, x-t+1, y-t+1] = f
            #     self.forces[1, x-t+1, y-t+1] = f
            #
            #     self.forces[0, x + t, y-t+1] = -f
            #     self.forces[1, x + t, y-t+1] = f
            #
            #     self.forces[0, x-t+1, y + t] = f
            #     self.forces[1, x-t+1, y + t] = -f
            #
            #     self.forces[0, x + t, y + t] = -f
            #     self.forces[1, x + t, y + t] = -f

        for [x, y] in self.sinkholes:
            self.forces[0, x - 3:x + 4, y - 3:y + 4] += fx
            self.forces[1, x - 3:x + 4, y - 3:y + 4] += fy



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
        # print(FFT(1,field0[:2,:]))
        self.field0cx = FFT(1, self.field0[0, :, :])
        self.field0cy = FFT(1, self.field0[1, :, :])
        # print(field0cx[0,5])
        # print(field0c.shape)
        # print(field0.shape)
        # field0[1,:] = FFT(1,field0[1,:])

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

                # field0c[0,i+(DIM+2)*  j].real = f*((1-x*x/r)*U[0] -x*y/r     *V[0]);
                # field0c[0,i+1+(DIM+2)*j].real = f*((1-x*x/r)*U[1] -x*y/r     *V[1]);
                # field0c[1,i+(DIM+2)*  j].imag = f*(-y*x/r*U[0]    +(1-y*y/r) *V[0]);
                # field0c[1,i+1+(DIM+2)*j].imag = f*(-y*x/r*U[1]    +(1-y*y/r) *V[1]);

        self.field0[0, :, :] = FFT(-1, self.field0cx)
        self.field0[1, :, :] = FFT(-1, self.field0cy)
        # field0[1,:DIM*DIM] = FFT(-1,field0[1,:])
        f = 1.0  # (DIM*DIM)
        for i in range(0, DIM):
            for j in range(0, DIM):
                self.field[:2, i, j] = f * self.field0[:2, i, j]

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
                # s = 1 - s
                # t = 1 - t
                self.field[-1, i, j] = (1 - s) * ((1 - t) * self.field0[-1, i0, j0] + t * self.field0[-1, i0, j1]) + s * (
                        (1 - t) * self.field0[-1, i1, j0] + t * self.field0[-1, i1, j1])


    def calc_divergence(self):
        DIM = self.DIM
        for i in range(0,DIM):
            for j in range(0,DIM):
                self.divfield[i,j] = -( self.field[0,i-1,j]-self.field[0,i,j] + self.field[0,i,j] - self.field[0,(i+1)%DIM,j] + self.field[1,i,j-1]-self.field[1,i,j] +  self.field[1,i,j] - self.field[1,i,(j+1)%DIM])
                self.divforces[i,j] = -(self.forces[0,i-1,j]-self.forces[0,i,j] + self.forces[0,(i+1)%DIM,j]-self.forces[0,i,j] + self.forces[1,i,j-1]-self.forces[1,i,j] + self.forces[1,i,(j+1)%DIM]- self.forces[1,i,j])


def clamp(x):
    return int(x) if x >= 0.0 else int(x - 1)


def FFT(direction, v):
    return np.fft.rfft2(v) if direction == 1 else np.fft.irfft2(v)


matrixx = 0.1 *np.array([[3, 2, -2, -3], [2, 1, -1, -2], [2, 1, -1, -2],[3, 2, -2, -3]])
matrixy = -0.1 * np.array([[-3, -2, -2, -3], [-2, -1, -1, -2], [2, 1, 1, 2],[3, 2, 2, 3]])
