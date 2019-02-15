# import pygame
import numpy as np
import sys
import math
# import pyfftw
# from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

# Visualization
frozen = False
winWidth = 500
winHeight = 500
color_dir = False

vec_scale = 1000
draw_smoke = False
draw_vecs = True
COLOR_BLACKWHITE = 0
COLOR_RAINBOW = 1
COLOR_BANDS = 2
scalar_col = 0

# Simulation
DIM = 50
dt = 0.4
visc = 0.01


def init_simulation():
    global field
    global field0
    global field0c
    global forces

    field = np.zeros((3, DIM, DIM))  # vx, vy, rho
    field0 = np.zeros((3, DIM, DIM))
    field0c = np.zeros((2, int(DIM * (DIM / 2 + 1))), dtype=np.complex_)  # vx0, vy0, rho0
    forces = np.zeros((2, DIM, DIM))  # fx, fy


def do_one_simulation_step():
    if not frozen:
        set_forces()
        solve()
        diffuse_matter()
        glutPostRedisplay()


def set_forces():
    field0[-1, :, :] = field[-1, :, :] * 0.9
    forces[:2, :, :] = forces[:2, :, :] * 0.85
    field0[:2, :, :] = forces[:2, :, :]


def clamp(x):
    return int(x) if x >= 0.0 else int(x - 1)


def FFT(direction, v):
    # varray = np.zeros((DIM+2,DIM))
    # for i in range(0,DIM+2):
    #     for j in range(0,DIM):
    #         varray[i,j] = v[i+(DIM+2)*j]
    # if direction == 1:
    #     trans = np.fft.rfftn(varray)
    # else:
    #     trans = np.fft.irfftn(varray)
    # print(varray)
    # print(trans)
    # vnew = np.zeros((DIM*(DIM+2)),dtype = np.complex_)
    # for i in range(0,DIM+2):
    #     for j in range(0,DIM):
    #         vnew[i+(DIM+2)*j] = trans[i,j]
    # return vnew
    return np.fft.rfft2(v) if direction == 1 else np.fft.irfft2(v)


def solve():
    field[:2, :, :] += dt * field0[:2, :, :]
    field0[:2, :, :] = field[:2, :, :]
    U = np.zeros(2)
    V = np.zeros(2)

    for i in range(0, DIM):
        for j in range(0, DIM):
            x = (0.5 + i) / DIM
            y = (0.5 + j) / DIM
            x0 = DIM * (x - dt * field0[0, i, j]) - 0.5
            y0 = DIM * (y - dt * field0[1, i, j]) - 0.5
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
            field[:2, i, j] = (1 - s) * ((1 - t) * field0[:2, i0, j0] + t * field0[:2, i0, j1]) + s * (
                        (1 - t) * field0[:2, i1, j0] + t * field0[:2, i1, j1])
    for i in range(0, DIM):
        for j in range(0, DIM):
            field0[:2, i, j] = field[:2, i, j]
    # print(FFT(1,field0[:2,:]))
    field0cx = FFT(1, field0[0, :, :])
    field0cy = FFT(1, field0[1, :, :])
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
            U[0] = field0cx[j, i].real
            V[0] = field0cy[j, i].real
            U[1] = field0cx[j, i].imag
            V[1] = field0cy[j, i].imag

            field0cx[j, i] = complex(f * ((1 - x * x / r) * U[0] - x * y / r * V[0]),
                                     f * ((1 - x * x / r) * U[1] - x * y / r * V[1]))
            field0cy[j, i] = complex((f * (-y * x / r * U[0] + (1 - y * y / r) * V[0])),
                                     (f * (-y * x / r * U[1] + (1 - y * y / r) * V[1])))

            # field0c[0,i+(DIM+2)*  j].real = f*((1-x*x/r)*U[0] -x*y/r     *V[0]);
            # field0c[0,i+1+(DIM+2)*j].real = f*((1-x*x/r)*U[1] -x*y/r     *V[1]);
            # field0c[1,i+(DIM+2)*  j].imag = f*(-y*x/r*U[0]    +(1-y*y/r) *V[0]);
            # field0c[1,i+1+(DIM+2)*j].imag = f*(-y*x/r*U[1]    +(1-y*y/r) *V[1]);

    field0[0, :, :] = FFT(-1, field0cx)
    field0[1, :, :] = FFT(-1, field0cy)
    # field0[1,:DIM*DIM] = FFT(-1,field0[1,:])
    f = 1.0  # (DIM*DIM)
    for i in range(0, DIM):
        for j in range(0, DIM):
            field[:2, i, j] = f * field0[:2, i, j]


def diffuse_matter():
    for i in range(0, DIM):
        for j in range(0, DIM):
            x = (0.5 + i) / DIM
            y = (0.5 + j) / DIM
            x0 = DIM * (x - dt * field[0, i, j]) - 0.5
            y0 = DIM * (y - dt * field[1, i, j]) - 0.5
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
            field[-1, i, j] = (1 - s) * ((1 - t) * field0[-1, i0, j0] + t * field0[-1, i0, j1]) + s * (
                        (1 - t) * field0[-1, i1, j0] + t * field0[-1, i1, j1])

### Visualization


def rainbow(value):
    dx = 0.8
    if value<0 :
        value=0
    if value>1 :
        value = 1
    value = (6-2*dx)*value+dx
    R = max(0.0,(3-np.fabs(value-4)-np.fabs(value-5))/2)
    G = max(0.0,(4-np.fabs(value-2)-np.fabs(value-4))/2)
    B = max(0.0,(3-np.fabs(value-1)-np.fabs(value-2))/2)
    return [R,G,B]


#set_colormap: Sets three different types of colormaps
def set_colormap(vy):
    RGB = np.zeros(3)

    if scalar_col==COLOR_BLACKWHITE:
        RGB.fill(vy)
    elif scalar_col==COLOR_RAINBOW:
       RGB = rainbow(vy)
    elif scalar_col==COLOR_BANDS:
        NLEVELS = 7
        vy *= NLEVELS
        vy = (int)(vy)
        vy/= NLEVELS
        rainbow(vy)
    glColor3f(RGB[0], RGB[1], RGB[2])


#direction_to_color: Set the current color by mapping a direction vector (x,y), using
#                    the color mapping method 'method'. If method==1, map the vector direction
#                    using a rainbow colormap. If method==0, simply use the white color
def direction_to_color(x, y, method):
    RGB = np.ones(3)
    if method:
        f = math.atan2(y, x) / 3.1415927 + 1
        r = f
        if r > 1:
            r = 2 - r
        g = f + .66667
        if g > 2:
            g -= 2
        if g > 1:
            g = 2 - g
        b = f + 2 * .66667
        if b > 2:
            b -= 2
        if b > 1:
            b = 2 - b
        RGB = [r, g, b]

    glColor3f(RGB[0], RGB[1], RGB[2])


def visualize():
    wn = winWidth / (DIM + 1)
    hn = winHeight / (DIM + 1)

    if draw_smoke:
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glBegin(GL_TRIANGLES)
        for i in range(0, DIM - 1):
            for j in range(0, DIM - 1):

                px0 = wn + i * wn
                py0 = hn + j * hn

                px1 = wn + i * wn
                py1 = hn + (j + 1) * hn
                #idx1 = ((j + 1) * DIM) + i;

                px2 = wn + (i + 1) * wn
                py2 = hn + (j + 1) * hn
                #dx2 = ((j + 1) * DIM) + (i + 1);

                px3 = wn + (i + 1) * wn
                py3 = hn + j * hn
                #idx3 = (j * DIM) + (i + 1);

                set_colormap(field[-1, i, j])
                glVertex2f(px0, py0)

                set_colormap(field[-1,i,j+1])
                glVertex2f(px1, py1)

                set_colormap(field[-1,i+1,j+1])
                glVertex2f(px2, py2)

                set_colormap(field[-1,i,j])
                glVertex2f(px0, py0)

                set_colormap(field[-1,i+1,j+1])
                glVertex2f(px2, py2)

                set_colormap(field[-1,i+1,j])
                glVertex2f(px3, py3)

        glEnd()

    if draw_vecs:
        glBegin(GL_LINES)
        for i in range(0, DIM):
            for j in range(0, DIM):
                direction_to_color(field[0, i, j], field[1, i, j], color_dir)
                glVertex2f(wn + i * wn, hn + j * hn)
                glVertex2f((wn + i * wn) + vec_scale * field[0, i, j], (hn + j * hn) + vec_scale * field[1, i, j])
                # print((wn + i*wn) + vec_scale *field[0,i,j] - wn + i*wn)
        glEnd()


# def static_vars(**kwargs):
#     def decorate(func):
#         for k in kwargs:
#             setattr(func, k, kwargs[k])
#         return func
#     return decorate
#
# # @static_vars(counter=0)
# # def foo():
# #     foo.counter += 1
# #     print "Counter is %d" % foo.counter
#
# @static_vars(lmx = 0)
# @static_vars(lmy=0)
def drag(mx, my):
    # lmx = 0
    # lmy = 0
    try:
        lmx = drag.lmx
        lmy = drag.lmy

    except AttributeError:
        drag.lmx = 0
        drag.lmy = 0
        lmx = drag.lmx
        lmy = drag.lmy

    xi = clamp((DIM + 1) * (mx / winWidth))
    yi = clamp((DIM + 1) * ((winHeight - my) / winHeight))
    X = int(xi)
    Y = int(yi)

    if X > (DIM - 1):
        X = DIM - 1
    if Y > (DIM - 1):
        Y = DIM - 1
    if X < 0:
        X = 0
    if Y < 0:
        Y = 0
    my = winHeight - my
    dx = mx - lmx
    dy = my - lmy
    length = np.sqrt(dx * dx + dy * dy)
    if not length == 0.0:
        dx = dx * (0.3 / length)
        dy = dy * (0.3/ length)
    forces[0, X, Y] += dx
    forces[1, X, Y] += dy
    field[-1, X, Y] = 10.0
    drag.lmx = mx
    drag.lmy = my
    # print([dx, dy])
    # print(forces)


def reshape(w, h):
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0.0, w, 0.0, h)
    winWidth = w
    winHeight = h


def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    visualize()
    glFlush()
    glutSwapBuffers()


def keyboard(key, x, y):
    ch = key.decode("utf-8")
    if ch == 't':
        global dt
        dt = dt - 0.001

    elif ch == 'T':
        dt += 0.001

    elif ch == 'c':
        global color_dir
        color_dir = not color_dir

    elif ch == 'S':
        global vec_scale
        vec_scale *= 1.2
    elif ch == 's':
        vec_scale *= 0.8
    elif ch == 'V':
        global visc
        visc *= 5
    elif ch == 'v':
        visc *= 0.2
    elif ch == 'x':
        global draw_smoke
        global draw_vecs
        draw_smoke = not draw_smoke
        if not draw_smoke:
            draw_vecs = True
    elif ch == 'y':
        draw_vecs = not draw_vecs
        if not draw_vecs:
            draw_smoke = True
    elif ch == 'm':
        global scalar_col
        scalar_col += 1
        if scalar_col > COLOR_BANDS:
            scalar_col = COLOR_BLACKWHITE
    elif ch == 'a':
        global frozen
        frozen = not frozen
    elif ch == 'q':
        sys.exit()


def main():
    print("Fluid Flow Simulation and Visualization\n")
    print("=======================================\n")
    print("Click and drag the mouse to steer the flow!\n")
    print("T/t:   increase/decrease simulation timestep\n")
    print("S/s:   increase/decrease hedgehog scaling\n")
    print("c:     toggle direction coloring on/off\n")
    print("V/vy:   increase decrease fluid viscosity\n")
    print("x:     toggle drawing matter on/off\n")
    print("y:     toggle drawing hedgehogs on/off\n")
    print("m:     toggle thru scalar coloring\n")
    print("a:     toggle the animation on/off\n")
    print("q:     quit\n\n")

    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(500, 500)
    glutCreateWindow("Real-time smoke simulation and visualization")
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutIdleFunc(do_one_simulation_step)
    glutKeyboardFunc(keyboard)
    glutMotionFunc(drag)
    init_simulation()
    glutMainLoop()


main()
# print(np.fft.rfft2(np.zeros((2,16))).size)
