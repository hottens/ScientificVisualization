# import pygame
import numpy as np
import sys
import math
# import pyfftw
# from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import QTimer, SIGNAL
from PyQt4.QtOpenGL import *


class MainWindow(QtGui.QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.widget = glWidget(self)

        self.button = QtGui.QPushButton('Test', self)

        mainLayout = QtGui.QHBoxLayout()
        mainLayout.addWidget(self.widget)
        mainLayout.addWidget(self.button)

        self.setLayout(mainLayout)

# Visualization
frozen = False
winWidth = 500
winHeight = 500
color_dir = False
color_mag_v = 0
colormap_type = 0
scale_velo_map = 5
NLEVELS = 2^256
levels = [2^256,20,10,5]
level = 0
hue = 0.0
sat = 1.0

scaling_factor_mag = 10
clamp_factor_mag = 0.02

magdir = True
vec_scale = 1000
draw_smoke = False
draw_vecs = True
COLOR_BLACKWHITE = 0
COLOR_RAINBOW = 1
COLOR_TWOTONE = 2
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
        #glutPostRedisplay()


def set_forces():
    field0[-1, :, :] = field[-1, :, :] * 0.9
    forces[:2, :, :] = forces[:2, :, :] * 0.85
    field0[:2, :, :] = forces[:2, :, :]


def clamp(x):
    return int(x) if x >= 0.0 else int(x - 1)


def FFT(direction, v):

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

    field0cx = FFT(1, field0[0, :, :])
    field0cy = FFT(1, field0[1, :, :])


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


    field0[0, :, :] = FFT(-1, field0cx)
    field0[1, :, :] = FFT(-1, field0cy)

    f = 1.0
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

            field[-1, i, j] = (1 - s) * ((1 - t) * field0[-1, i0, j0] + t * field0[-1, i0, j1]) + s * (
                        (1 - t) * field0[-1, i1, j0] + t * field0[-1, i1, j1])

### Visualization



def hsv2rgb(h,s,v):
    hint = int(h*6)
    frac = 6*hint
    lx = v * (1 - s)
    ly = v * (1 - s*frac)
    lz = v * ( 1 - s * ( 1 - frac))
    if hint == 6:
        RGB = [v, lz, lx]
    elif hint == 1:
        RGB = [ly,v,lx]
    elif hint == 2:
        RGB = [lx,v,lz]
    elif hint == 3:
        RGB = [lx,ly,v]
    elif hint == 4:
        RGB = [lz, lx, v]
    else:
        RGB = [v,lx,ly]
    return RGB

def rgb2hsv(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    v = mx
    if mx > 0.00001:
        s = df/mx
    else:
        s = 0
    if s == 0:
        h = 0
    else:
        if r == mx:
            h = (g-b)/df
        elif g == mx:
            h = 2 + (b-r)/df
        else:
            h = 4 + (r-g)/df
        h = h/6
        if h < 0:
            h += 1

    return h, s, v



def rainbow(cv):
    dx = 0.8
    if cv<0 :
        cv=0
    if cv>1 :
        cv = 1


    cv = (6-2*dx)*cv+dx
    R = max(0.0,(3-np.fabs(cv-4)-np.fabs(cv-5))/2)
    G = max(0.0,(4-np.fabs(cv-2)-np.fabs(cv-4))/2)
    B = max(0.0,(3-np.fabs(cv-1)-np.fabs(cv-2))/2)

    if not hue == 0:
        [h,s,v] = rgb2hsv(R,G,B)

        h = (h+hue)%1
    #     R,G,B = hsv2rgb(h,s,v)
    R = sat*R
    G = sat*G
    B = sat*B

    return [R,G,B]

def twotone(value):
    c1= [255/256,255/256,51/256]
    c2 =  [0.0,51/256,255/256]


    if value<0 :
        value=0
    if value>1 :
        value = 1

    R = value * (c1[0]-c2[0]) + c2[0]
    G = value * (c1[1]-c2[1]) + c2[1]
    B = value * (c1[2]-c2[2]) + c2[2]
    if not hue == 0:
        [h,s,v] = rgb2hsv(R,G,B)
        h = (h+hue)%1
        R,G,B = hsv2rgb(h,s,v)
    R = sat*R
    G = sat*G
    B = sat*B
    return [R,G,B]


#set_colormap: Sets three different types of colormaps
def set_colormap(vy):
    RGB = np.zeros(3)

    if not NLEVELS == 2^256:
        vy = vy * NLEVELS
        vy = int(vy)
        vy = vy / NLEVELS

    if scalar_col==COLOR_BLACKWHITE:
        RGB.fill(vy)
    elif scalar_col==COLOR_RAINBOW:
       RGB = rainbow(vy)
    elif scalar_col == COLOR_TWOTONE:
        RGB = twotone(vy)
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

def magnitude_to_color(x,y, colormaptype):
    RGB = np.ones(3)
    mag = np.sqrt(x*x + y*y)

    mag = scaling_factor_mag * mag + clamp_factor_mag
    if mag > 1:
        mag = 1
    if mag < 0:
        mag = 0
    if colormaptype == 0:
        RGB = [mag,mag,mag]
    elif colormaptype == 1:
        RGB = rainbow(mag)
    elif colormaptype == 2:
        RGB = twotone(mag)

    glColor3f(RGB[0],RGB[1],RGB[2])




def visualize():
    wn = winWidth / (DIM + 1)
    hn = winHeight / (DIM + 1)


    if draw_smoke:
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        colormap = np.zeros((3,50,50))
        if colormap_type == 0:
            colormap = field[-1,:,:]
        elif colormap_type == 1:
            colormap = scale_velo_map * np.sqrt(field[0,:,:]*field[0,:,:] + field[1,:,:]*field[1,:,:])
        elif colormap_type == 2:
            colormap = np.sqrt(forces[0,:,:]*forces[0,:,:] + forces[1,:,:]*forces[1,:,:])

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

                set_colormap(colormap[i, j])
                glVertex2f(px0, py0)

                set_colormap(colormap[i,j+1])
                glVertex2f(px1, py1)

                set_colormap(colormap[i+1,j+1])
                glVertex2f(px2, py2)

                set_colormap(colormap[i,j])
                glVertex2f(px0, py0)

                set_colormap(colormap[i+1,j+1])
                glVertex2f(px2, py2)

                set_colormap(colormap[i+1,j])
                glVertex2f(px3, py3)

        glEnd()

    if draw_vecs:
        glBegin(GL_LINES)
        for i in range(0, DIM):
            for j in range(0, DIM):
                if magdir:
                    magnitude_to_color(field[0, i, j], field[1, i, j], color_mag_v)
                else:
                    direction_to_color(field[0, i, j], field[1, i, j], color_dir)
                glVertex2f(wn + i * wn, hn + j * hn)
                glVertex2f((wn + i * wn) + vec_scale * field[0, i, j], (hn + j * hn) + vec_scale * field[1, i, j])
                # print((wn + i*wn) + vec_scale *field[0,i,j] - wn + i*wn)
        glEnd()


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
    #glutSwapBuffers()


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

    elif ch == 'C':
        global color_mag_v
        color_mag_v += 1
        if color_mag_v > 2:
            color_mag_v = 0
    elif ch == 'z':
        global magdir
        magdir = not magdir
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
        if scalar_col > COLOR_TWOTONE:
            scalar_col = COLOR_BLACKWHITE
    elif ch == 'n':
        global colormap_type
        colormap_type += 1
        if colormap_type > 2:
            colormap_type = 0
    elif ch == 'a':
        global frozen
        frozen = not frozen
    elif ch == 'h':
        global hue
        hue += 1/6
        if hue > 1.0:
            hue = 0
    elif ch == 'l':
        global NLEVELS
        global level
        level += 1
        if level > 3:
            level = 0
        NLEVELS = levels[level]

    elif ch == 'q':
        sys.exit()


class glWidget(QGLWidget):

    def __init__(self, parent):
        QGLWidget.__init__(self, parent)
        self.setMinimumSize(500, 500)
        QtGui.qApp.installEventFilter(self)
        self.timer = QTimer()
        self.timer.timeout.connect(self.on_timer)
        self.timer.start(1000)

    def on_timer(self):
        print(forces)
        self.repaint_grid()
        self.update()

    def repaint_grid(self):
        do_one_simulation_step()
        display()

    def mousePressEvent(self, event):
        self.__mousePressPos = None
        self.__mouseMovePos = None
        if event.button() == QtCore.Qt.LeftButton:
            self.__mousePressPos = event.globalPos()
            self.__mouseMovePos = event.globalPos()

        super(glWidget, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() == QtCore.Qt.LeftButton:
            # adjust offset from clicked point to origin of widget
            currPos = self.mapToGlobal(self.pos())
            globalPos = event.globalPos()
            diff = globalPos - self.__mouseMovePos
            newPos = self.mapFromGlobal(currPos + diff)
            #self.move(newPos)
            drag(diff.x(),diff.y())
            print("drag")

            self.__mouseMovePos = globalPos

        super(glWidget, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.__mousePressPos is not None:
            moved = event.globalPos() - self.__mousePressPos
            if moved.manhattanLength() > 3:
                event.ignore()
                return

        super(glWidget, self).mouseReleaseEvent(event)

    def eventFilter(self, source, event):
        if event.type() == QtCore.QEvent.KeyPress:
            print('KeyPress: %s [%r]' % (event.key(), source))
        return super(glWidget, self).eventFilter(source, event)

    def initializeGL(self):
        init_simulation()
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, 1.33, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)


if __name__ == '__main__':
    print("Fluid Flow Simulation and Visualization\n")
    print("=======================================\n")
    print("Click and drag the mouse to steer the flow!\n")
    print("T/t:   increase/decrease simulation timestep\n")
    print("S/s:   increase/decrease hedgehog scaling\n")
    print("c:     toggle direction coloring on/off\n")
    print("C:     toggle magnitude velocity coloring on/off\n")
    print("z:     velocity coloring based on direction/magnitude\n")
    print("V/v:   increase decrease fluid viscosity\n")
    print("x:     toggle drawing matter on/off\n")
    print("y:     toggle drawing hedgehogs on/off\n")
    print("m:     toggle thru scalar coloring\n")
    print("n:     toggle thru what is displayed on the colormap (density, velocity, force)\n")
    print("a:     toggle the animation on/off\n")
    print("l:     change number of colors in colormap (2**256, 20, 10, 5) \n ")
    print("h:     change hue \n")

    print("q:     quit\n\n")

    app = QtGui.QApplication(['Yo'])
    window = MainWindow()
    window.show()
    app.exec_()
