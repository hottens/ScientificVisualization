import numpy as np
import sys
import math
import queue
import simulation
import socket
import time
from fluid_actions import Action
import pygame
from ctypes import *
from threading import Thread
from PIL import Image
import random


from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

# Server
serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serversocket.bind(('localhost', 8089))
serversocket.listen(5)  # become a server socket, maximum 5 connections
keep_connection = True

# Simulation
DIM = 50
sim = simulation.Simulation(DIM)

# actions
q = queue.Queue(maxsize=20)

# Visualization
frozen = False
winWidth = 500
winHeight = 500
color_mag_v = 0

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






COLOR_BLACKWHITE = 0
COLOR_RAINBOW = 1
COLOR_TWOTONE = 2
COLOR_WHITE = 3
scalar_col = 0




### Options
# vectors:          none, hedgehogs, triangles, arrows
# vector_color:     white, direction_to_color, magnitude_to_color
# color_schemes:    bw, twotone, rainbow
# color_map_type:   diffusion, direction / magnitude, force, divergence
# isolines:         no, yest
# n_isolines


color_dict = {'Field': {'nlevels': 256, 'scale': 1.0, 'color_scheme': COLOR_BLACKWHITE, 'show': False, 'clamp_min': 0.0, 'clamp_max':1.0, 'datatype': 0}, \
 'Iso': {'nlevels': 256, 'scale': 1.0, 'color_scheme': COLOR_WHITE, 'show': True, 'clamp_min': 0.0, 'clamp_max':1.0, 'iso_min': 0.7, 'iso_max':1.0, 'iso_n': 1},\
  'Vector': {'nlevels': 256, 'scale': 1.0, 'color_scheme': COLOR_RAINBOW, 'show': True, 'clamp_min': 0.0, 'clamp_max':0.1, 'n_glyphs': 16, 'draw_glyphs': 2, 'col_mag': 1, 'vec_scale': 5}}


colormap_vect = np.zeros((256,3))
colormap_field = np.zeros((256,3))
colormap_iso = np.zeros((256,3))



def createAndCompileShader(type, source):
    shader = glCreateShader(type)
    glShaderSource(shader, source)
    glCompileShader(shader)

    # get "compile status" - glCompileShader will not fail with
    # an exception in case of syntax errors

    result = glGetShaderiv(shader, GL_COMPILE_STATUS)

    if (result != 1):  # shader didn't compile
        raise Exception("Couldn't compile shader\nShader compilation Log:\n" + glGetShaderInfoLog(shader))
    return shader

vertex_shader = createAndCompileShader(GL_VERTEX_SHADER, """
varying vec3 v;
varying vec3 N;

void main(void)
{

   v = gl_ModelViewMatrix * gl_Vertex;
   N = gl_NormalMatrix * gl_Normal;

   gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;

}
""");


fragment_shader = createAndCompileShader(GL_FRAGMENT_SHADER, """
varying vec3 N;
varying vec3 v;

void main(void)
{
   vec3 L = gl_LightSource[0].position.xyz-v;

   // "Lambert's law"? (see notes)
   // Rather: faces will appear dimmer when struck in an acute angle
   // distance attenuation

   float Idiff = max(dot(normalize(L),N),0.0)*pow(length(L),-2.0);

   gl_FragColor = vec4(0.5,0,0.5,1.0)+ // purple
                  vec4(1.0,1.0,1.0,1.0)*Idiff; // diffuse reflection
}
""");

program = glCreateProgram()
glAttachShader(program, vertex_shader)
glAttachShader(program, fragment_shader)
glLinkProgram(program)

try:
    glUseProgram(program)
except OpenGL.error.GLError:
    print(glGetProgramInfoLog(program))
    raise
t = 0


### Visualization

def isolines():
    nlevels = color_dict['Iso']['nlevels']
    clamp_min = color_dict['Iso']['clamp_min']
    clamp_max = color_dict['Iso']['clamp_max']
    max = color_dict['Iso']['iso_max']
    min = color_dict['Iso']['iso_min']
    n = color_dict['Iso']['iso_n']
    if max < min:
        return
    if max == min:
        n = 1
    vallist = []
    if n == 1:
        vallist = [min]
    else:
        for i in range(0, n):
            vallist += [min + i * (max - min) / (n - 1)]

    glBegin(GL_LINES)
    for val in vallist:
        threshold_image = sim.field[-1, :, :] > val
        if val > clamp_max:
            val = clamp_max
        elif val < clamp_min:
            val = clamp_min

        cv = colormap_iso[int(round(val/(clamp_max-clamp_min) * (nlevels-1)))]

        for i in range(0, DIM - 1):
            for j in range(0, DIM - 1):
                bincode_list = ['0', '0', '0', '0']
                if threshold_image[i, j]:
                    bincode_list[0] = '1'
                if threshold_image[i + 1, j]:
                    bincode_list[1] = '1'
                if threshold_image[i + 1, j + 1]:
                    bincode_list[2] = '1'
                if threshold_image[i, j + 1]:
                    bincode_list[3] = '1'

                bincode = "".join(bincode_list)

                if int(bincode, 2) == 1 or int(bincode, 2) == 14:
                    x1 = i + (sim.field[-1, i, j + 1] - val) / (sim.field[-1, i, j + 1] - sim.field[-1, i + 1, j + 1])
                    y1 = j + 1
                    x2 = i
                    y2 = j + (sim.field[-1, i, j] - val) / (sim.field[-1, i, j] - sim.field[-1, i, j + 1])
                elif int(bincode, 2) == 2 or int(bincode, 2) == 13:
                    x1 = i + (sim.field[-1, i, j + 1] - val) / (sim.field[-1, i, j + 1] - sim.field[-1, i + 1, j + 1])
                    y1 = j + 1
                    x2 = i + 1
                    y2 = j + (sim.field[-1, i + 1, j] - val) / (sim.field[-1, i + 1, j] - sim.field[-1, i + 1, j + 1])
                elif int(bincode, 2) == 3 or int(bincode, 2) == 12:
                    x1 = i
                    y1 = j + (sim.field[-1, i, j] - val) / (sim.field[-1, i, j] - sim.field[-1, i, j + 1])
                    x2 = i + 1
                    y2 = j + (sim.field[-1, i + 1, j] - val) / (sim.field[-1, i + 1, j] - sim.field[-1, i + 1, j + 1])
                elif int(bincode, 2) == 4 or int(bincode, 2) == 11 or int(bincode, 2) == 10:
                    x1 = i + (sim.field[-1, i, j] - val) / (sim.field[-1, i, j] - sim.field[-1, i + 1, j])
                    y1 = j
                    x2 = i + 1
                    y2 = j + (sim.field[-1, i + 1, j] - val) / (sim.field[-1, i + 1, j] - sim.field[-1, i + 1, j + 1])
                elif int(bincode, 2) == 6 or int(bincode, 2) == 9:
                    x1 = i + (sim.field[-1, i, j] - val) / (sim.field[-1, i, j] - sim.field[-1, i + 1, j])
                    y1 = j
                    x2 = i + (sim.field[-1, i, j + 1] - val) / (sim.field[-1, i, j + 1] - sim.field[-1, i + 1, j + 1])
                    y2 = j + 1
                elif int(bincode, 2) == 7 or int(bincode, 2) == 8 or int(bincode, 2) == 5:
                    x1 = i + (sim.field[-1, i, j] - val) / (sim.field[-1, i, j] - sim.field[-1, i + 1, j])
                    y1 = j
                    x2 = i
                    y2 = j + (sim.field[-1, i, j] - val) / (sim.field[-1, i, j] - sim.field[-1, i, j + 1])
                if int(bincode, 2) == 5:
                    x12 = i + (sim.field[-1, i, j + 1] - val) / (sim.field[-1, i, j + 1] - sim.field[-1, i + 1, j + 1])
                    y12 = j + 1
                    x22 = i + 1
                    y22 = j + (sim.field[-1, i + 1, j] - val) / (sim.field[-1, i + 1, j] - sim.field[-1, i + 1, j + 1])
                elif int(bincode, 2) == 10:
                    x12 = i + (sim.field[-1, i, j + 1] - val) / (sim.field[-1, i, j + 1] - sim.field[-1, i + 1, j + 1])
                    y12 = j + 1
                    x22 = i
                    y22 = j + (sim.field[-1, i, j] - val) / (sim.field[-1, i, j] - sim.field[-1, i, j + 1])

                if not int(bincode, 2) == 0 or int(bincode, 2) == 15:
                    glColor3f(cv[0], cv[1], cv[2])
                    glVertex2f((x1 / (50 / 2)) - 1, (y1 / (50 / 1.8)) - 0.8)
                    glVertex2f((x2 / (50 / 2)) - 1, (y2 / (50 / 1.8)) - 0.8)

                if int(bincode, 2) == 5 or int(bincode, 2) == 10:
                    glColor3f(cv[0], cv[1], cv[2])
                    glVertex2f((x12 / (50 / 2)) - 1, (y12 / (50 / 1.8)) - 0.8)
                    glVertex2f((x22 / (50 / 2)) - 1, (y22 / (50 / 1.8)) - 0.8)
    glEnd()


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




def bw(cv,scale):
    RGB = np.zeros(3)
    global clamp_color
    # if cv < clamp_color[0]:
    #     cv = clamp_color[0]
    # if cv > clamp_color[1]:
    #     cv = clamp_color[1]
    cv = cv**scale
    RGB = np.array([cv, cv, cv])
    return RGB


# return color from rainbow colormap based on a value
def rainbow(cv,scale):
    dx = 0.8
    # global clamp_color
    # if cv < clamp_color[0]:
    #     cv = clamp_color[0]
    # if cv > clamp_color[1]:
    #     cv = clamp_color[1]
    cv = cv**scale
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

    return np.array([R, G, B])


# return color from twotone colormap based on a value
def twotone(value,scale):
    c1 = [255 / 256, 255 / 256, 51 / 256]
    c2 = [0.0, 51 / 256, 255 / 256]
    global clamp_color
    # if value < clamp_color[0]:
    #     value = clamp_color[0]
    # if value > clamp_color[1]:
    #     value = clamp_color[1]
    value = cv**scale


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

    return np.array([R, G, B])


def change_colormap(type):
    global colormap_field
    global colormap_iso
    global colormap_vect
    nlevels = color_dict[type]['nlevels']
    scale = color_dict[type]['scale']
    color_scheme = color_dict[type]['color_scheme']
    colormap = np.zeros((nlevels, 3))
    for i in range(0,nlevels):
        if color_scheme == COLOR_BLACKWHITE:
            colormap[i,:] = bw(i/nlevels, scale)
        elif color_scheme == COLOR_RAINBOW:
            colormap[i,:] = rainbow(i/nlevels, scale)
        elif color_scheme == COLOR_TWOTONE:
            colormap[i,:] = twotone(i/nlevels, scale)
        elif color_scheme == COLOR_WHITE:
            colormap[i,:] = np.ones((1,3))

    if type == 'Field':
        colormap_field = colormap
    elif type == 'Iso':
        colormap_iso = colormap
    elif type == 'Vector':
        colormap_vect = colormap


def makecolormap(colormaptobe):
    colormap = np.zeros((50, 50, 3))
    nlevels = color_dict['Field']['nlevels']
    clamp_min = color_dict['Field']['clamp_min']
    clamp_max = color_dict['Field']['clamp_max']
    for i in range(0, DIM):
        for j in range(0, DIM):
            val = colormaptobe[i,j]
            if val < clamp_min:
                val = clamp_min
            elif val > clamp_max:
                val = clamp_max
            val = (val-clamp_min)/(clamp_max-clamp_min) * (nlevels-1)
            val = int(round(val))

            colormap[i, j, :] = colormap_field[val,:]
            # print(colormap[i,j,:])
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


def blackcolors():
    return [0] * 43218


### Determine the representation of the colors
def vis_color():
    colormaptobe = np.zeros((50, 50))
    colormap_type = color_dict['Field']['datatype']
    # Density
    if colormap_type == 0:
        colormaptobe = sim.field[-1, :, :]

    # Direction and Magnitude
    elif colormap_type == 1:
        colormaptobe = scale_velo_map * np.sqrt(
            sim.field[0, :, :] * sim.field[0, :, :] + sim.field[1, :, :] * sim.field[1, :, :])

    # Forces
    elif colormap_type == 2:
        colormaptobe = np.sqrt(sim.forces[0, :, :] * sim.forces[0, :, :] + sim.forces[1, :, :] * sim.forces[1, :, :])

    # Divergence
    elif colormap_type == 3:

        colormaptobe = 50 * sim.divfield[:, :] + 0.5

    global colors
    colors = makecolormap(colormaptobe)


# returns the vertices needed for printing the colormap
def makevertices():
    v = []
    for i in range(49):
        for j in range(49):
            p0 = [i / (49 / 2) - 1, j / (49 / 1.8) - 0.8, -1]
            p1 = [i / (49 / 2) - 1, (j + 1) / (49 / 1.8) - 0.8, -1]
            p2 = [(i + 1) / (49 / 2) - 1, (j + 1) / (49 / 1.8) - 0.8, -1]
            p3 = [(i + 1) / (49 / 2) - 1, j / (49 / 1.8) - 0.8, -1]
            v += p0 + p1 + p2 + p0 + p2 + p3
    v = np.array(v)
    # v = v / (49 / 2) - 1
    v = v.tolist()
    return v


def makelegend():
    vertices_leg = []
    colors_leg = []
    length = color_dict['Field']['nlevels']

    for i in range(length):
        p0 = [i / (length / 2) - 1, -1.0, 0]
        p1 = [(i + 1) / (length / 2) - 1, -1.0, 0]
        p2 = [(i + 1) / (length / 2) - 1, -0.8, 0]
        p3 = [i / (length / 2) - 1, -0.8, 0]
        vertices_leg += p0 + p1 + p2 + p0 + p2 + p3
        colval = colormap_field[i]
        colval = colval.tolist()
        colors_leg += colval * 6
    vertices_leg = np.array(vertices_leg)
    vertices_leg = vertices_leg.tolist()

    # colors_leg = np.array(colors_leg)

    leg_vbo = glGenBuffers(1)
    colleg_vbo = glGenBuffers(1)

    glEnableClientState(GL_COLOR_ARRAY)
    glBindBuffer(GL_ARRAY_BUFFER, leg_vbo)
    glBufferData(GL_ARRAY_BUFFER, len(vertices_leg) * 4, (c_float * len(vertices_leg))(*vertices_leg), GL_STATIC_DRAW)
    glVertexPointer(3, GL_FLOAT, 0, None)

    glBindBuffer(GL_ARRAY_BUFFER, colleg_vbo)
    glBufferData(GL_ARRAY_BUFFER, len(colors_leg) * 4, (c_float * len(colors_leg))(*colors_leg), GL_STATIC_DRAW)

    glColorPointer(3, GL_FLOAT, 0, None)
    glDrawArrays(GL_TRIANGLES, 0, 6*length)


########## VECTOR COLORING
# direction_to_color: Set the current color by mapping a direction vector (x,y), using
#                    the color mapping method 'method'. If method==1, map the vector direction
#                    using a rainbow colormap. If method==0, simply use the white color
def direction_to_color(x, y):
    RGB = np.ones(3)
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

    mag = scaling_factor_mag * mag
    clamp_min = color_dict['Vector']['clamp_min']
    clamp_max = color_dict['Vector']['clamp_max']
    nlevels = color_dict['Vector']['nlevels']
    if mag > clamp_max:
        mag = clamp_max
    if mag < clamp_min:
        mag = clamp_min
    RGB = colormap_vect[int(round(mag/(clamp_max-clamp_min)*(nlevels-1)))]
    # if colormaptype == 0:
    #     RGB = [mag, mag, mag]
    # elif colormaptype == 1:
    #     RGB = rainbow(mag)
    # elif colormaptype == 2:
    #     RGB = twotone(mag)

    return RGB


# functions that draws cones as glyphs
def drawGlyph(x, y, vx, vy, size, color):
    size += 5
    glBegin(GL_TRIANGLES)
    glColor3f(color[0]*0.7, color[1]*0.7, color[2]*0.7)
    # a = np.array([x + vx, y + vy, 0])
    # b = np.array([x - 10 / DIM * vy, y + 10 / DIM * vx,0])
    # c = np.array([x, y,0.8])
    # d = np.array([x + 10 / DIM * vy, y - 10 / DIM * vx,0])
    # norm1 = np.cross((b - a),(c-a))
    # norm1 = norm1/len(norm1)
    # norm2 = np.cross((c-a),(d-a))
    # norm2 = norm2/len(norm2)
    # glNormal3f(norm1[0], norm1[1], norm1[2])
    # glNormal3f(0.5, -0.5, 0.5)

    glVertex3f(x + vx, y + vy, 0)
    glVertex3f(x - 10 / DIM * vy, y + 10 / DIM * vx,0)
    glColor3f(color[0]*1.2, color[1]*1.2, color[2]*1.2)
    glVertex3f(x, y,0.8)
    # glNormal3f(norm2[0], norm2[1], norm2[2])
    # glNormal3f(0.5, 0.5, 0.5)
    # glColor3f(color[0], color[1], color[2])
    glColor3f(color[0]*0.4, color[1]*0.4, color[2]*0.4)
    glVertex3f(x + vx, y + vy, 0)
    glVertex3f(x + 10 / DIM * vy, y - 10 / DIM * vx,0)
    glColor3f(color[0], color[1], color[2])
    glVertex3f(x, y,0.8)
    glEnd()


# function that draws arrows as glyphs
def drawArrow(x, y, vx, vy, size, color):
    size += 5
    glBegin(GL_LINES)
    glColor3f(color[0], color[1], color[2])
    glVertex2f(x + vx, y + vy)
    glVertex2f(x, y)
    glEnd()
    glBegin(GL_TRIANGLES)
    glColor3f(color[0], color[1], color[2])

    glVertex2f((x + 0.5 * vx) - 2 * (size / DIM) * vy, (y + 0.5 * vy) + 2 * (size / DIM) * vx)
    glColor3f(color[0]-0.2, color[1]-0.2, color[2]-0.2)
    glVertex2f(x + vx, y + vy)
    glVertex2f((x + 0.5 * vx) + 2 * (size / DIM) * vy, (y + 0.5 * vy) - 2 * (size / DIM) * vx)
    glEnd()


##### USER INPUT
def placeSinkHole(mx, my):
    xi = simulation.clamp((DIM + 1) * (mx / winWidth))
    yi = simulation.clamp((DIM + 1) * ((winHeight - my) / winHeight))
    X = int(xi)
    Y = int(yi)

    sim.sinkholes += [[X, Y]]

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
    global color_dict
    global color_mag_v
    global clamp_color
    global scalar_col
    global frozen
    global hue
    global NLEVELS
    global level
    global scale

    a = message.split(':')
    action = a[0]

    if action == Action.DT_DOWN.name:
        sim.dt -= 0.001
    elif action == Action.DT_UP.name:
        sim.dt += 0.001
    elif action == Action.SET_DT.name:
        sim.dt = float(a[1])
    elif action == Action.COLOR_MAG_CHANGE.name:
        color_mag_v += 1
        if color_mag_v > 2:
            color_mag_v = 0
    elif action == Action.MAG_DIR.name:
        color_dict['Vector']['col_mag'] += 1
        if color_dict['Vector']['col_mag'] > 2:
            color_dict['Vector']['col_mag'] = 0
    elif action == Action.VEC_SCALE_UP.name:
        color_dict['Vector']['vec_scale'] *= 1.2
    elif action == Action.VEC_SCALE_DOWN.name:
        color_dict['Vector']['vec_scale'] *= 0.8
    elif action == Action.VISC_UP.name:
        sim.visc *= 5
    elif action == Action.VISC_DOWN.name:
        sim.visc *= 0.2
    elif action == Action.DRAW_SMOKE.name:
        color_dict['Field']['show'] = not color_dict['Field']['show']
    elif action == Action.DRAW_VECS.name:
        color_dict['Vector']['show'] = not color_dict['Vector']['show']
    elif action == Action.DRAW_ISO.name:
        color_dict['Iso']['show'] = not color_dict['Iso']['show']
    elif action == Action.GLYPH_CHANGE.name:
        color_dict['Vector']['draw_glyphs'] += 1
        if color_dict['Vector']['draw_glyphs'] > 4:
            color_dict['Vector']['draw_glyphs'] = 0
    elif action == Action.SCALAR_COLOR_CHANGE.name:
        color_dict['Field']['color_scheme'] += 1
        if color_dict['Field']['color_scheme'] > COLOR_TWOTONE:
            color_dict['Field']['color_scheme'] = COLOR_BLACKWHITE
    elif action == Action.COLOR_MAG_BLACK.name:
        color_dict['Vector']['color_scheme'] = COLOR_BLACKWHITE
        change_colormap('Vector')
    elif action == Action.COLOR_MAG_RAINBOW.name:
        color_dict['Vector']['color_scheme'] = COLOR_RAINBOW
        change_colormap('Vector')
    elif action == Action.COLOR_MAG_TWOTONE.name:
        color_dict['Vector']['color_scheme'] = COLOR_TWOTONE
        change_colormap('Vector')
    elif action == Action.COLORMAP_CHANGE.name:
        color_dict['Field']['datatype'] += 1
        if color_dict['Field']['datatype'] > 3:
            color_dict['Field']['datatype'] = 0
    elif action == Action.FREEZE.name:
        frozen = not frozen
    elif action == Action.CLAMP_COLOR_MIN_UP.name:
        clamp_color[0] = min(clamp_color[0] + 0.1, clamp_color[1] - 0.1)
    elif action == Action.CLAMP_COLOR_MAX_DOWN.name:
        clamp_color[0] = max(clamp_color[0] - 0.1, 0)
    elif action == Action.CLAMP_COLOR_MAX_UP.name:
        clamp_color[1] = max(clamp_color[1] + 0.1, 1)
    elif action == Action.CLAMP_COLOR_MAX_DOWN.name:
        clamp_color[1] = min(clamp_color[1] - 0.1, clamp_color[0] + 0.1)
    elif action == Action.CHANGE_HUE.name:
        hue += 1 / 6
        if hue >= 1.0:
            hue = 0
    elif action == Action.CHANGE_LEVELS.name:
        level += 1
        if level > 3:
            level = 0
        NLEVELS = levels[level]
    elif action == Action.SET_NLEVELS_FIELD.name:
        color_dict['Field']['nlevels'] = int(a[1])
        change_colormap('Field')
    elif action == Action.SET_NLEVELS_ISO.name:
        color_dict['Iso']['nlevels'] = int(a[1])
        change_colormap('Iso')
    elif action == Action.SET_NLEVELS_VECTOR.name:
        color_dict['Vector']['nlevels'] = int(a[1])
        change_colormap('Vector')
    elif action == Action.GLYPH_CHANGE_N.name:
        color_dict['Vector']['n_glyphs'] += 5
        if color_dict['Vector']['n_glyphs'] > 50:
            color_dict['Vector']['n_glyphs'] = 5
    elif action == Action.SET_SCALE_FIELD.name:
        color_dict['Field']['scale'] = float(a[1])
        change_colormap('Field')
    elif action == Action.SET_SCALE_VECTOR.name:
        color_dict['Vector']['scale'] = float(a[1])
        change_colormap('Vector')
    elif action == Action.SET_SCALE_ISO.name:
        color_dict['Iso']['scale'] = float(a[1])
        change_colormap('Iso')
    elif action == Action.CHANGE_ISO_COL.name:
        color_dict['Iso']['color_scheme'] += 1
        if color_dict['Iso']['color_scheme'] > 4:
            color_dict['Iso']['color_scheme'] = 0
    elif action == Action.SET_ISO_MIN.name:
        color_dict['Iso']['iso_min'] = float(a[1])
    elif action == Action.SET_ISO_MAX.name:
        color_dict['Iso']['iso_max'] = float(a[1])
    elif action == Action.SET_ISO_N.name:
        color_dict['Iso']['iso_n'] = int(a[1])
    elif action == Action.COLOR_ISO_BLACK.name:
        color_dict['Iso']['color_scheme'] = 0
    elif action == Action.COLOR_ISO_RAINBOW.name:
        color_dict['Iso']['color_scheme'] = 1
    elif action == Action.COLOR_ISO_TWOTONE.name:
        color_dict['Iso']['color_scheme'] = 2
    elif action == Action.COLOR_ISO_WHITE.name:
        color_dict['Iso']['color_scheme'] = 3
    elif action == Action.COLORMAP_TYPE_DENSITY.name:
        color_dict['Field']['datatype'] = 0
    elif action == Action.COLORMAP_TYPE_VELOCITY.name:
        color_dict['Field']['datatype'] = 1
    elif action == Action.COLORMAP_TYPE_FORCES.name:
        color_dict['Field']['datatype'] = 2
    elif action == Action.COLORMAP_TYPE_DIVERGENCE.name:
        color_dict['Field']['datatype'] = 3
    elif action == Action.QUIT.name:
        global keep_connection
        keep_connection = False
        time.sleep(2)
        sys.exit()


# function that gets the keyboard input, which is used for controlling the parameters
#       of the simulation.
def keyboard(key):
    global q
    if key == pygame.K_t:
        q.put(Action.DT_DOWN.name)
    elif key == pygame.K_y:
        q.put(Action.DT_UP.name)

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
    elif key == pygame.K_7:
        q.put(Action.CHANGE_ISO_COL.name)
    elif key == pygame.K_q:
        q.put(Action.QUIT.name)


def getGuiInput():
    global q
    global keep_connection
    # connect to socket
    while keep_connection:
        connection, address = serversocket.accept()
        buf = connection.recv(64)
        if len(buf) > 0:
            q.put(buf.decode('utf_8'))
            print(buf.decode('utf_8'))
        connection.close()


def interpolateVelocity(x, y):
    x_floor = int(np.floor(x))
    x_ceil  = int(np.ceil(x))
    y_floor = int(np.floor(y))
    y_ceil  = int(np.ceil(y))

    x = x-x_floor
    y = y-y_floor

    #print("xf: {}, xc: {}, yf: {}, yc: {}".format(x_floor, x_ceil, y_floor, y_ceil))

    vx = sim.field[0, x_floor, y_floor] * (1 - x) * (1 - y) + \
         sim.field[0, x_ceil,  y_floor] * x       * (1 - y) + \
         sim.field[0, x_floor, y_ceil ] * (1 - x) * y       + \
         sim.field[0, x_ceil,  y_ceil ] * x * y

    vy = sim.field[1, x_floor, y_floor] * (1 - x) * (1 - y) + \
         sim.field[1, x_ceil, y_floor] * x * (1 - y) + \
         sim.field[1, x_floor, y_ceil] * (1 - x) * y + \
         sim.field[1, x_ceil, y_ceil] * x * y

    return [vx, vy]

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

    thread = Thread(target=getGuiInput)
    thread.start()

    change_colormap('Field')
    change_colormap('Iso')
    change_colormap('Vector')


    vertices = makevertices()
    global colors
    colors = makecolormap(sim.field[-1, :, :])

    vertices_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vertices_vbo)
    glBufferData(GL_ARRAY_BUFFER, len(vertices) * 4, (c_float * len(vertices))(*vertices), GL_STATIC_DRAW)

    colors_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, colors_vbo)
    glBufferData(GL_ARRAY_BUFFER, len(colors) * 4, (c_float * len(colors))(*colors), GL_STATIC_DRAW)


    # glColor3f(1,1,1)

    running = True
    while running:



        clock.tick(60)
        global dragbool
        global keep_connection
        if not keep_connection:
            thread.join()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    dragbool = True
                elif event.button == 3:
                    mx, my = event.pos
                    placeSinkHole(mx, my)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragbool = False
            if event.type == pygame.KEYDOWN:
                keyboard(event.key)
        if dragbool:
            mx, my = event.pos
            drag(mx, my)







        while not q.empty():
            performAction(q.get())
            # print(q.get())

        sim.do_one_simulation_step(frozen)
        vis_color()

        glEnableClientState(GL_COLOR_ARRAY)

        glBindBuffer(GL_ARRAY_BUFFER, vertices_vbo)
        glVertexPointer(3, GL_FLOAT, 0, None)
        if not color_dict['Field']['show']:
            colors = blackcolors()
        glBindBuffer(GL_ARRAY_BUFFER, colors_vbo)
        glBufferData(GL_ARRAY_BUFFER, len(colors) * 4, (c_float * len(colors))(*colors), GL_STATIC_DRAW)
        glColorPointer(3, GL_FLOAT, 0, None)

        glDrawArrays(GL_TRIANGLES, 0, 43218)


        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        # gluPerspective(90, 1, 0.01, 1)
        # gluLookAt(0,0, 1,0, 0, 0, 0, 1, 0)
        # glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)




        draw_glyphs = color_dict['Vector']['draw_glyphs']
        n_glyphs = color_dict['Vector']['n_glyphs']
        vec_scale = color_dict['Vector']['vec_scale']
        show_vecs = color_dict['Vector']['show']
        if show_vecs:



            glNewList(1, GL_COMPILE)
            if draw_glyphs == 1 :

                glBegin(GL_LINES)
                step = DIM / n_glyphs
                for i in range(0, n_glyphs):
                    for j in range(0, n_glyphs):

                        x = round(i * step)
                        y = round(j * step)
                        color = np.ones(3)
                        if color_dict['Vector']['col_mag'] == 1:
                            color = magnitude_to_color(sim.field[0, x, y], sim.field[1, x, y], color_mag_v)
                        elif color_dict['Vector']['col_mag'] == 2:
                            color = direction_to_color(sim.field[0, x, y], sim.field[1, x, y])
                        glColor3f(color[0], color[1], color[2])

                        glVertex2f((((i + 0.5) * step / (49 / 2)) - 1), (((j + 0.5) * step / (49 / 1.8)) - 0.8))
                        glVertex2f((((i + 0.5) * step / (49 / 2)) - 1) + vec_scale * sim.field[0, x, y],
                                   (((j + 0.5) * step / (49 / 1.8)) - 0.8) + vec_scale * sim.field[1, x, y])
                glEnd()

            if draw_glyphs >= 2:
                step = DIM / n_glyphs

                for i in range(n_glyphs):
                    for j in range(n_glyphs):

                        x = i * step
                        y = j * step
                        vx = step * sim.field[0, round(x), round(y)]
                        vy = step * sim.field[1, round(x), round(y)]
                        x2 = (i + 0.5) * step / ((DIM - 1) / 2) - 1
                        y2 = (j + 0.5) * step / ((DIM - 1) / 1.8) - 0.8

                        color = np.ones(3)
                        if color_dict['Vector']['col_mag'] == 1:
                            color = magnitude_to_color(sim.field[0, round(x), round(y)], sim.field[1, round(x), round(y)],
                                                       color_mag_v)
                        elif color_dict['Vector']['col_mag'] == 2:
                            color = direction_to_color(sim.field[0, round(x), round(y)], sim.field[1, round(x), round(y)])

                        size = sim.field[-1, round(x), round(y)]
                        if draw_glyphs == 2:
                            drawGlyph(x2, y2, vx, vy, size, color)
                        elif draw_glyphs == 3:
                            drawArrow(x2, y2, vx, vy, size, color)

                        else:
                            glBegin(GL_LINES)
                            T = 5
                            x_d = x2
                            y_d = y2
                            for t in range(T):
                                if x > 49 or y > 49 or x < 0 or y < 0:
                                    break
                                #print("x: {}, y: {}".format(x, y))
                                [vx, vy] = interpolateVelocity(x, y)
                                # vx = sim.field[0, round(x), round(y)]
                                # vy = sim.field[1, round(x), round(y)]
                                v_l = np.sqrt(vx*vx + vy*vy)
                                x_t = x + vx / (v_l)
                                y_t = y + vy / (v_l)
                                color = np.ones(3)
                                if color_dict['Vector']['col_mag'] == 1:
                                    color = magnitude_to_color(vx, vy, color_mag_v)
                                elif color_dict['Vector']['col_mag'] == 2:
                                    color = direction_to_color(vx, vy)
                                glColor3f(color[0], color[1], color[2])

                                glVertex2f(x_d, y_d)
                                x_d += vx/(v_l*49)
                                y_d += vy/(v_l*49)
                                glVertex2f(x_d, y_d)
                                x = x_t
                                y = y_t
                            glEnd()


            glEndList()

        if color_dict['Iso']['show']:
            isolines()
        # glMatrixMode(GL_MODELVIEW)




        # pass data to fragment shader

        # ld = [1,0.8,1]
        # glLightfv(GL_LIGHT0, GL_AMBIENT, [1,1,1])
        # glLightfv(GL_LIGHT0, GL_DIFFUSE, [1, 1, 1]);
        # glLightfv(GL_LIGHT0, GL_POSITION,[ld[0], ld[1], ld[2]]);


        # glColor3f(1,1,1)

        # glLoadIdentity()
        # glPushMatrix()
        if show_vecs:
            glCallList(1)
        # glPopMatrix()
        makelegend()
        pygame.display.flip()


pygame.init()
screen = pygame.display.set_mode((winWidth, winHeight + 55), pygame.OPENGL | pygame.DOUBLEBUF, 24)

# glEnable(GL_DEPTH_TEST)
# glShadeModel(GL_FLAT)
# glEnable(GL_LIGHTING)
# glEnable(GL_LIGHT0)
# glViewport(0, 0, winWidth, winHeight + 55)
# glClearColor(0.0, 0.5, 0.5, 1.0)
glEnableClientState(GL_VERTEX_ARRAY)
# glEnableClientState(GL_COLOR_ARRAY)
main()
