import numpy as np
import sys
import math
import queue
import simulation
import socket
from fluid_actions import Action
import pygame
from OpenGL.GL import *
from ctypes import *
from threading import Thread

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

# Server
serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serversocket.bind(('localhost', 8089))
serversocket.listen(5)  # become a server socket, maximum 5 connections

# Simulation
DIM = 50
sim = simulation.Simulation(DIM)

# actions
q = queue.Queue(maxsize=20)

# Visualization
frozen = False
winWidth = 500
winHeight = 500
color_dir = False
color_mag_v = 0
colormap_type = 0
scale_velo_map = 5
NLEVELS = 256 ^ 3
levels = [256 ^ 3, 20, 10, 5]
level = 0
hue = 0.0
sat = 1.0
dragbool = False

scaling_factor_mag = 2
clamp_factor_mag = 0.02

colors = []

clamp_color = [0, 1]

magdir = False
vec_scale = 5
draw_smoke = True
draw_vecs = False
draw_glyphs = 1
n_glyphs = 16
COLOR_BLACKWHITE = 0
COLOR_RAINBOW = 1
COLOR_TWOTONE = 2
scalar_col = 0
scale = 1.0


### Visualization

####### COLORMAPPING

# converse hsv to rgb coloring
def hsv2rgb(h, s, v):
    hint = int(h * 6)
    frac = 6 * hint
    lx = v * (1 - s)
    ly = v * (1 - s * frac)
    lz = v * (1 - s * (1 - frac))
    if hint == 6:
        RGB = [v, lz, lx]
    elif hint == 1:
        RGB = [ly, v, lx]
    elif hint == 2:
        RGB = [lx, v, lz]
    elif hint == 3:
        RGB = [lx, ly, v]
    elif hint == 4:
        RGB = [lz, lx, v]
    else:
        RGB = [v, lx, ly]
    return RGB

# converse rgb to hsv coloring
def rgb2hsv(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn
    v = mx
    if mx > 0.00001:
        s = df / mx
    else:
        s = 0
    if s == 0:
        h = 0
    else:
        if r == mx:
            h = (g - b) / df
        elif g == mx:
            h = 2 + (b - r) / df
        else:
            h = 4 + (r - g) / df
        h = h / 6
        if h < 0:
            h += 1

    return h, s, v

def scalecolor(cv):
    return cv**scale


def bw(cv):
    RGB = np.zeros(3)
    global clamp_color
    if cv < clamp_color[0]:
        cv = clamp_color[0]
    if cv > clamp_color[1]:
        cv = clamp_color[1]
    cv = scalecolor(cv)
    RGB = [cv, cv, cv]
    return RGB



# return color from rainbow colormap based on a value
def rainbow(cv):
    dx = 0.8
    global clamp_color
    if cv < clamp_color[0]:
        cv = clamp_color[0]
    if cv > clamp_color[1]:
        cv = clamp_color[1]
    cv = scalecolor(cv)
    cv = (6 - 2 * dx) * cv + dx
    R = max(0.0, (3 - np.fabs(cv - 4) - np.fabs(cv - 5)) / 2)
    G = max(0.0, (4 - np.fabs(cv - 2) - np.fabs(cv - 4)) / 2)
    B = max(0.0, (3 - np.fabs(cv - 1) - np.fabs(cv - 2)) / 2)

    if not hue == 0:
        [h, s, v] = rgb2hsv(R, G, B)

        h = (h + hue) % 1
        R, G, B = hsv2rgb(h, s, v)
    R = sat * R
    G = sat * G
    B = sat * B

    return [R, G, B]

# return color from twotone colormap based on a value
def twotone(value):
    c1 = [255 / 256, 255 / 256, 51 / 256]
    c2 = [0.0, 51 / 256, 255 / 256]
    global clamp_color
    if value < clamp_color[0]:
        value = clamp_color[0]
    if value > clamp_color[1]:
        value = clamp_color[1]
    value = scalecolor(value)
    # nog scalen!!

    R = value * (c1[0] - c2[0]) + c2[0]
    G = value * (c1[1] - c2[1]) + c2[1]
    B = value * (c1[2] - c2[2]) + c2[2]
    if not hue == 0:
        [h, s, v] = rgb2hsv(R, G, B)
        h = (h + hue) % 1
        R, G, B = hsv2rgb(h, s, v)
    R = sat * R
    G = sat * G
    B = sat * B

    return [R, G, B]


# set_colormap: Sets three different types of colormaps
def set_colormap(vy):
    RGB = np.zeros(3)
    global scalar_col
    if not NLEVELS == 256^3:
        vy = vy * NLEVELS
        vy = int(vy)
        vy = vy / NLEVELS

    if scalar_col == COLOR_BLACKWHITE:
        RGB = bw(vy)
    elif scalar_col == COLOR_RAINBOW:
        RGB = rainbow(vy)
    elif scalar_col == COLOR_TWOTONE:
        RGB = twotone(vy)
    return RGB


def makecolormap(colormaptobe):
    colormap = np.zeros((50, 50, 3))
    for i in range(0, DIM):
        for j in range(0, DIM):
            colormap[i, j, :] = set_colormap(colormaptobe[i, j])
    c = []
    for x in range(0, DIM - 1):
        for y in range(0, DIM - 1):
            c += [colormap[x, y, 0], colormap[x, y, 1], colormap[x, y, 2]]
            c += [colormap[x, y + 1, 0], colormap[x, y + 1, 1], colormap[x, y + 1, 2]]
            c += [colormap[x + 1, y + 1, 0], colormap[x + 1, y + 1, 1], colormap[x + 1, y + 1, 2]]
            c += [colormap[x, y, 0], colormap[x, y, 1], colormap[x, y, 2]]
            c += [colormap[x + 1, y + 1, 0], colormap[x + 1, y + 1, 1], colormap[x + 1, y + 1, 2]]
            c += [colormap[x + 1, y, 0], colormap[x + 1, y, 1], colormap[x + 1, y, 2]]
    return c


def blackcolormap():
    return [0]*43218


def vis_color():
    colormaptobe = np.zeros((50, 50))
    if colormap_type == 0:
        colormaptobe = sim.field[-1, :, :]
    elif colormap_type == 1:
        colormaptobe = scale_velo_map * np.sqrt(sim.field[0, :, :] * sim.field[0, :, :] + sim.field[1, :, :] * sim.field[1, :, :])
    elif colormap_type == 2:
        colormaptobe = np.sqrt(sim.forces[0, :, :] * sim.forces[0, :, :] + sim.forces[1, :, :] * sim.forces[1, :, :])
    global colors
    colors = makecolormap(colormaptobe)


# returns the vertices needed for printing the colormap
def makevertices():
    v = []
    for i in range(49):
        for j in range(49):
            p0 = [i/ (49 / 2) - 1, j/ (49 / 1.8) - 0.8, 0]
            p1 = [i/ (49 / 2) - 1, (j + 1)/ (49 / 1.8) - 0.8, 0]
            p2 = [(i + 1)/ (49 / 2) - 1, (j + 1)/ (49 / 1.8) - 0.8, 0]
            p3 = [(i + 1)/ (49 / 2) - 1, j/ (49 / 1.8) - 0.8, 0]
            v += p0 + p1 + p2 + p0 + p2 + p3
    v = np.array(v)
    # v = v / (49 / 2) - 1
    v = v.tolist()
    return v

def makelegend():
    vertices_leg = []
    colors_leg = []

    for i in range(499):
        p0 = [i    /(499/2) - 1,-1.0, 0]
        p1 = [(i+1)/(499/2) - 1,-1.0, 0]
        p2 = [(i+1)/(499/2) - 1,-0.8, 0]
        p3 = [i    /(499/2) - 1,-0.8, 0]
        vertices_leg += p0 + p1 + p2 + p0 + p2 + p3
        colval = set_colormap(i/499)
        # colval = colval.tolist()
        colors_leg += colval*6
    vertices_leg = np.array(vertices_leg)
    vertices_leg = vertices_leg.tolist()

    # colors_leg = np.array(colors_leg)





    leg_vbo = glGenBuffers(1)
    colleg_vbo = glGenBuffers(1)


    glEnableClientState(GL_COLOR_ARRAY)
    glBindBuffer(GL_ARRAY_BUFFER, leg_vbo)
    glBufferData(GL_ARRAY_BUFFER, len(vertices_leg) * 4, (c_float * len(vertices_leg))(*vertices_leg), GL_STATIC_DRAW)
    glVertexPointer(3, GL_FLOAT,0,None)


    glBindBuffer(GL_ARRAY_BUFFER, colleg_vbo)
    glBufferData(GL_ARRAY_BUFFER, len(colors_leg) * 4, (c_float * len(colors_leg))(*colors_leg), GL_STATIC_DRAW)


    glColorPointer(3,GL_FLOAT,0,None)
    glDrawArrays(GL_TRIANGLES,0,2994)




########## VECTOR COLORING
# direction_to_color: Set the current color by mapping a direction vector (x,y), using
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

    return RGB


# returns color depending on magnitude of vector. Depending on colormaptype the type
#       of coloring is decided (0->white, 1->rainbow, 2-> twotone)
def magnitude_to_color(x, y, colormaptype):
    RGB = np.ones(3)
    mag = np.sqrt(x * x + y * y)

    mag = scaling_factor_mag * mag + clamp_factor_mag
    if mag > 1:
        mag = 1
    if mag < 0:
        mag = 0
    if colormaptype == 0:
        RGB = [mag, mag, mag]
    elif colormaptype == 1:
        RGB = rainbow(mag)
    elif colormaptype == 2:
        RGB = twotone(mag)

    return RGB


# functions that draws cones as glyphs
def drawGlyph(x, y, vx, vy, size, color):
    size += 5
    glBegin(GL_TRIANGLES)
    glColor3f(color[0], color[1], color[2])
    glVertex2f(x + vx, y + vy)
    glVertex2f(x - 10/DIM * vy, y + 10/DIM * vx)
    glVertex2f(x + 10/DIM * vy, y - 10/DIM * vx)
    glEnd()


# function that draws arrows as glyphs
def drawArrow(x, y, vx, vy, size, color):
    size+=5
    glBegin(GL_LINES)
    glColor3f(color[0], color[1], color[2])
    glVertex2f(x + vx, y + vy)
    glVertex2f(x, y)
    glEnd()
    glBegin(GL_TRIANGLES)
    glColor3f(color[0], color[1], color[2])
    glVertex2f(x + vx, y + vy)
    glVertex2f((x+0.5*vx) - 2*(size / DIM) * vy, (y+0.5*vy) + 2*(size / DIM) * vx)
    glVertex2f((x+0.5*vx) + 2*(size / DIM) * vy, (y+0.5*vy) - 2*(size / DIM) * vx)
    glEnd()


##### USER INPUT


# gets the drag movement of the mouse and changes the simulation values
#       according to these movements
def drag(mx, my):
    my = my
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

    xi = simulation.clamp((DIM + 1) * (mx / winWidth))
    yi = simulation.clamp((DIM + 1) * ((winHeight - my) / winHeight))
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
        dy = dy * (0.3 / length)
    sim.forces[0, X, Y] += dx
    sim.forces[1, X, Y] += dy
    sim.field[-1, X, Y] = 10.0
    drag.lmx = mx
    drag.lmy = my


def performAction(message):
    global vec_scale
    global color_dir
    global color_mag_v
    global magdir
    global clamp_color
    global draw_smoke
    global draw_glyphs
    global colormap_type
    global scalar_col
    global frozen
    global hue
    global NLEVELS
    global level
    global n_glyphs
    global scale

    a = message.split(':')
    action = a[0]

    if action == Action.DT_DOWN.name:
        sim.dt -= 0.001
    elif action == Action.DT_UP.name:
        sim.dt += 0.001
    elif action == Action.SET_DT.name:
        sim.dt = float(a[1])
    elif action == Action.COLOR_DIR.name:
        color_dir = not color_dir
    elif action == Action.COLOR_MAG_CHANGE.name:
        color_mag_v += 1
        if color_mag_v > 2:
            color_mag_v = 0
    elif action == Action.SET_COLOR_MAG.name:
        color_mag_v = float(['vector coloring bw', 'vector coloring rainbow', 'vector coloring twotone'].index(a[1]))
    elif action == Action.MAG_DIR.name:
        magdir = not magdir
    elif action == Action.VEC_SCALE_UP.name:
        vec_scale *= 1.2
    elif action == Action.VEC_SCALE_DOWN.name:
        vec_scale *= 0.8
    elif action == Action.VISC_UP.name:
        sim.visc *= 5
    elif action == Action.VISC_DOWN.name:
        sim.visc *= 0.2
    elif action == Action.DRAW_SMOKE.name:
        draw_smoke = not draw_smoke
    elif action == Action.GLYPH_CHANGE.name:
        draw_glyphs += 1
        if draw_glyphs > 3:
            draw_glyphs = 0
    elif action == Action.SCALAR_COLOR_CHANGE.name:
        scalar_col += 1
        if scalar_col > COLOR_TWOTONE:
            scalar_col = COLOR_BLACKWHITE
    elif action == Action.COLORMAP_CHANGE.name:
        colormap_type += 1
        if colormap_type > 2:
            colormap_type = 0
    elif action == Action.FREEZE.name:
        frozen = not frozen
    elif action == Action.CLAMP_COLOR_MIN_UP.name:
        clamp_color[0] = min(clamp_color[0]+0.1, clamp_color[1]-0.1)
    elif action == Action.CLAMP_COLOR_MAX_DOWN.name:
        clamp_color[0] = max(clamp_color[0]-0.1, 0)
    elif action == Action.CLAMP_COLOR_MAX_UP.name:
        clamp_color[1] = max(clamp_color[1]+0.1, 1)
    elif action == Action.CLAMP_COLOR_MAX_DOWN.name:
        clamp_color[1] = min(clamp_color[1]-0.1, clamp_color[0]+0.1)
    elif action == Action.CHANGE_HUE.name:
        hue += 1 / 6
        if hue >= 1.0:
            hue = 0
    elif action == Action.CHANGE_LEVELS.name:
        level += 1
        if level > 3:
            level = 0
        NLEVELS = levels[level]
    elif action == Action.GLYPH_CHANGE_N.name:
        n_glyphs += 5
        if n_glyphs > 50:
            n_glyphs = 5
    elif action == Action.SET_SCALE.name:
        scale = float(a[1])
    elif action == Action.QUIT.name:
        sys.exit()


# function that gets the keyboard input, which is used for controlling the parameters
#       of the simulation.
def keyboard(key):
    global q
    if key == pygame.K_t:
        q.put(Action.DT_DOWN.name)
    elif key == pygame.K_y:
        q.put(Action.DT_UP.name)
    elif key == pygame.K_c:
        q.put(Action.COLOR_DIR.name)
    elif key == pygame.K_v:
        q.put(Action.COLOR_MAG_CHANGE.name)
    elif key == pygame.K_m:
        q.put(Action.MAG_DIR.name)
    elif key == pygame.K_a:
        q.put(Action.VEC_SCALE_UP.name)
    elif key == pygame.K_s:
        q.put(Action.VEC_SCALE_DOWN.name)
    elif key == pygame.K_z:
        q.put(Action.VISC_UP.name)
    elif key == pygame.K_x:
        q.put(Action.VISC_DOWN.name)
    elif key == pygame.K_n:
        q.put(Action.DRAW_SMOKE.name)
    elif key == pygame.K_u:
        q.put(Action.GLYPH_CHANGE.name)
    elif key == pygame.K_i:
        q.put(Action.SCALAR_COLOR_CHANGE.name)
    elif key == pygame.K_j:
        q.put(Action.COLORMAP_CHANGE.name)
    elif key == pygame.K_f:
        q.put(Action.FREEZE.name)
    elif key == pygame.K_1:
        q.put(Action.CLAMP_COLOR_MIN_UP.name)
    elif key == pygame.K_2:
        q.put(Action.CLAMP_COLOR_MIN_DOWN.name)
    elif key == pygame.K_3:
        q.put(Action.CLAMP_COLOR_MAX_UP.name)
    elif key == pygame.K_4:
        q.put(Action.CLAMP_COLOR_MAX_DOWN.name)
    elif key == pygame.K_h:
        q.put(Action.CHANGE_HUE.name)
    elif key == pygame.K_l:
        q.put(Action.CHANGE_LEVELS.name)
    elif key == pygame.K_g:
        q.put(Action.GLYPH_CHANGE_N.name)
    elif key == pygame.K_q:
        q.put(Action.QUIT.name)


def getGuiInput():
    global q
    # connect to socket
    while True:
        connection, address = serversocket.accept()
        buf = connection.recv(64)
        if len(buf) > 0:
            q.put(buf.decode('utf_8'))
            print(buf.decode('utf_8'))
        connection.close()


def main():
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
    print("l:     change number of colors in colormap (256**3, 20, 10, 5) \n ")
    print("h:     change hue \n")

    print("q:     quit\n\n")

    clock = pygame.time.Clock()

    thread = Thread(target = getGuiInput)
    thread.start()

    vertices = makevertices()
    global colors
    colors = makecolormap(sim.field[-1, :, :])

    vertices_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vertices_vbo)
    glBufferData(GL_ARRAY_BUFFER, len(vertices) * 4, (c_float * len(vertices))(*vertices), GL_STATIC_DRAW)

    colors_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, colors_vbo)
    glBufferData(GL_ARRAY_BUFFER, len(colors) * 4, (c_float * len(colors))(*colors), GL_STATIC_DRAW)

    running = True
    while running:
        clock.tick(60)
        global dragbool
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    dragbool = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragbool = False
            if event.type == pygame.KEYDOWN:
                keyboard(event.key)
        if dragbool:
            mx, my = event.pos
            drag(mx, my)

        glClear(GL_COLOR_BUFFER_BIT)

        while not q.empty():
            performAction(q.get())
            #print(q.get())

        sim.do_one_simulation_step(frozen)
        vis_color()

        glEnableClientState(GL_COLOR_ARRAY)

        glBindBuffer(GL_ARRAY_BUFFER, vertices_vbo)
        glVertexPointer(3, GL_FLOAT, 0, None)
        if not draw_smoke:
            colors = blackcolormap()
        glBindBuffer(GL_ARRAY_BUFFER, colors_vbo)
        glBufferData(GL_ARRAY_BUFFER, len(colors) * 4, (c_float * len(colors))(*colors), GL_STATIC_DRAW)
        glColorPointer(3, GL_FLOAT, 0, None)

        glDrawArrays(GL_TRIANGLES, 0, 43218)

        makelegend()
        if draw_glyphs == 1:
            glBegin(GL_LINES)
            step = DIM/n_glyphs
            for i in range(0, n_glyphs):
                for j in range(0, n_glyphs):

                    x = round(i*step)
                    y = round(j*step)
                    color = np.ones(3)
                    if magdir:
                        color = magnitude_to_color(sim.field[0, x, y], sim.field[1, x, y], color_mag_v)
                    else:
                        color = direction_to_color(sim.field[0, x, y], sim.field[1, x, y], color_dir)
                    glColor3f(color[0], color[1], color[2])

                    glVertex2f((((i+0.5)*step / (49 / 2)) - 1), (((j+0.5)*step / (49 / 1.8)) - 0.8))
                    glVertex2f((((i+0.5)*step / (49 / 2)) - 1) + vec_scale * sim.field[0, x, y],
                               ((((j+0.5)*step / (49 / 1.8)) - 0.8)) + vec_scale * sim.field[1, x, y])
            glEnd()

        if draw_glyphs >= 2:
            step = DIM/n_glyphs
            for i in range(n_glyphs):
                for j in range(n_glyphs):

                    x = i*step
                    y = j*step
                    vx = step * sim.field[0, round(x), round(y)]
                    vy = step * sim.field[1, round(x), round(y)]
                    x2 = (i+0.5)*step / ((DIM - 1) / 2) - 1
                    y2 = (j+0.5)*step / ((DIM - 1) / 1.8) - 0.8

                    color = np.ones(3)
                    if magdir:
                        color = magnitude_to_color(sim.field[0, round(x), round(y)], sim.field[1, round(x), round(y)], color_mag_v)
                    else:
                        color = direction_to_color(sim.field[0, round(x), round(y)], sim.field[1, round(x), round(y)], color_dir)

                    size = sim.field[-1, round(x), round(y)]
                    if draw_glyphs == 2:
                        drawGlyph(x2, y2, vx, vy, size, color)
                    else:
                        drawArrow(x2, y2, vx, vy, size, color)

        pygame.display.flip()


pygame.init()
screen = pygame.display.set_mode((winWidth, winHeight+55), pygame.OPENGL | pygame.DOUBLEBUF, 24)
glViewport(0, 0, winWidth, winHeight+55)
glClearColor(0.0, 0.5, 0.5, 1.0)
glEnableClientState(GL_VERTEX_ARRAY)


main()
