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


scale_velo_map = 5

dragbool = False


colors = []





# constants for the different types of color schemes
COLOR_BLACKWHITE = 0
COLOR_RAINBOW = 1
COLOR_TWOTONE = 2
COLOR_WHITE = 3


# dictionary that holds most of the parameters for each of the three types of elements that
# can be shown.
#       GENERAL PARAMETERS
# 'nlevels'     :   holds the number of colors that are shown
# 'scale'       :   holds the scaling factor for the colors
# 'color_scheme':   holds in what color scheme the given type is shown
# 'show'        :   will this element be shown?
# 'clamp_min'   :   minimum clamp value for this elements color
# 'clamp_max'   :   maximum clamp value for this elements color
# 'hue'         :   hue of the shown colors
# 'sat'         :   saturation of the shown colors
#       FIELD PARAMETERS
# '3d'          :   holds whether the Visualization is shown in 3d or note
# 'heightscale' :   holds the scaling factor for the height
# 'heightfactor':   holds the factor with which the height is multiplied
# 'datatype'    :   holds whether density (=0), velocity (=1), forces (=2) or
#                   divergence (=3) is shown
#       ISO PARAMETERS
# 'iso_min'     :   holds value for lowest isoline
# 'iso_max'     :   holds value for highest isoline
# 'iso_n'       :   holds how many isolines are shown (if 'iso_n' = 1, the value
#                   of the isoline is 'iso_min')
#       VECTOR PARAMETERS
# 'n_glyphs'    :   holds the square root number of displayed glyphs (i.e. if
#                   'n_glyphs' = 9, the number of displayed glyphs is 81)
# 'draw_glyphs' :   holds which types of glyphs are shown: hedgehogs(=1), cones (=2)
#                   arrows (=3), streamlines (=4), streamtubes (=5)
# 'col_mag'     :   holds whether we show directional or magnitude colorcoding
# 'vec_scale'   :   holds the value with which the vector magnitude is multiplied
# 'velocity'    :   holds whether we show the velocity (=True) or the forcefield (=False)
# 'slinelength' :   holds the number of timesteps for which the streamline is calculated
# 'displacement':   holds whether the glyphs are shown in a grid, or whether they are
#                   slightly displaced. (corresponds to option random seeding in GUI)

parameter_dict = {'Field': {'nlevels': 256, 'scale': 1.0, 'color_scheme': COLOR_BLACKWHITE,
                        'show': True, 'clamp_min': 0.0, 'clamp_max':1.0, 'datatype': 0,
                        '3d': False, 'heightscale': 1.0, 'heightfactor': 0.05, 'hue':0,'sat':1.0},
              'Iso': {'nlevels': 256, 'scale': 1.0, 'color_scheme': COLOR_WHITE,
                      'show': False, 'clamp_min': 0.0, 'clamp_max':1.0, 'iso_min': 0.7, 'iso_max':1.0, 'iso_n': 1,
                      'hue': 0,'sat':1.0},
              'Vector': {'nlevels': 256, 'scale': 1.0, 'color_scheme': COLOR_WHITE,
                         'show': True, 'clamp_min': 0.0, 'clamp_max':1.0, 'n_glyphs': 16, 'draw_glyphs': 2,
                         'col_mag': 1, 'vec_scale': 5, 'hue':0,'sat':1.0, 'velocity': True, 'slinelength': 5,
                         'displacement': False}}


colormap_vect = np.zeros((256,3))
colormap_field = np.zeros((256,3))
colormap_iso = np.zeros((256,3))
displacement = np.zeros((2,50,50))




### Visualization

def isolines():
    # load in variables
    nlevels = parameter_dict['Iso']['nlevels']
    clamp_min = parameter_dict['Iso']['clamp_min']
    clamp_max = parameter_dict['Iso']['clamp_max']
    max = parameter_dict['Iso']['iso_max']
    min = parameter_dict['Iso']['iso_min']
    n = parameter_dict['Iso']['iso_n']
    hf = parameter_dict['Field']['heightfactor']
    hs = parameter_dict['Field']['heightscale']
    # make sure the max of the iso lines is higher than the min
    if max < min:
        return
    if max == min:
        n = 1
    # create values for the isolines
    vallist = []
    if n == 1:
        vallist = [min]
    else:
        for i in range(0, n):
            vallist += [min + i * (max - min) / (n - 1)]

    # draw isolines using marching squares
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

                # draw the isolines
                if not int(bincode, 2) == 0 or int(bincode, 2) == 15:
                    glColor3f(cv[0], cv[1], cv[2])
                    glVertex3f((x1 / (49 / 2)) - 1, (y1 / (49 / 1.8)) - 0.8, (hf * val * 1.3)**hs)
                    glVertex3f((x2 / (49 / 2)) - 1, (y2 / (49 / 1.8)) - 0.8, (hf * val * 1.3)**hs)

                if int(bincode, 2) == 5 or int(bincode, 2) == 10:
                    glColor3f(cv[0], cv[1], cv[2])
                    glVertex3f((x12 / (49 / 2)) - 1, (y12 / (49 / 1.8)) - 0.8, (hf * val *1.3)**hs)
                    glVertex3f((x22 / (49 / 2)) - 1, (y22 / (49 / 1.8)) - 0.8, (hf * val *1.3)**hs)
    glEnd()


####### COLORMAPPING

# converse hsv to rgb coloring
def hsv2rgb(h, s, v):
    hint = int(h * 6)
    frac = (h * 6) - hint
    lx = v * (1 - s)
    ly = v * (1 - s * frac)
    lz = v * (1 - s * (1 - frac))
    hint = hint % 6
    if hint == 0:
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
            h = 6 + (g - b) / df
        elif g == mx:
            h = 2 + (b - r) / df
        else:
            h = 4 + (r - g) / df
        h = h % 6
        h = h /6

    return h, s, v

# return color for black and white color map
def bw(cv,scale):
    RGB = np.zeros(3)
    cv = cv**scale
    RGB = np.array([cv, cv, cv])
    return RGB


# return color for rainbow colormap based on a value
def rainbow(cv,scale,hue,sat):
    dx = 0.8
    cv = cv**scale
    cv = (6 - 2 * dx) * cv + dx
    R = max(0.0, (3 - np.fabs(cv - 4) - np.fabs(cv - 5)) / 2)
    G = max(0.0, (4 - np.fabs(cv - 2) - np.fabs(cv - 4)) / 2)
    B = max(0.0, (3 - np.fabs(cv - 1) - np.fabs(cv - 2)) / 2)

    # include hue and sat from user
    [h, s, v] = rgb2hsv(R, G, B)
    h = (h + (hue/6)) % 1
    R, G, B = hsv2rgb(h, sat, v)
    return np.array([R, G, B])


# return color from twotone colormap based on a value
def twotone(cv,scale,hue,sat):
    c1 = [0.9, 0.9, 0.0]
    c2 = [0.0, 0.0, 0.9]

    value = cv**scale
# interpolate between colors
    R = (value * (c1[0] - c2[0]) + c2[0])
    G = (value * (c1[1] - c2[1]) + c2[1])
    B = (value * (c1[2] - c2[2]) + c2[2])

# include hue from user
    [h, s, v] = rgb2hsv(R, G, B)
    h = (h + (hue/6)) % 1
    R, G, B = hsv2rgb(h, s, v)

    return np.array([R, G, B])


# this function changes the color map when one of the parameters has option_changed
# type holds either 'Field' 'Vector' or 'Iso', depending on the colormap that
# should be changed
def change_colormap(type):
    global colormap_field
    global colormap_iso
    global colormap_vect
    # retrieve parameters
    nlevels = parameter_dict[type]['nlevels']
    scale = parameter_dict[type]['scale']
    color_scheme = parameter_dict[type]['color_scheme']
    hue = parameter_dict[type]['hue']
    sat = parameter_dict[type]['sat']

    # create correct colormap
    colormap = np.zeros((nlevels, 3))
    for i in range(0,nlevels):
        if color_scheme == COLOR_BLACKWHITE:
            colormap[i,:] = bw(i/(nlevels-1), scale)
        elif color_scheme == COLOR_RAINBOW:
            colormap[i,:] = rainbow(i/(nlevels-1), scale,hue,sat)
        elif color_scheme == COLOR_TWOTONE:
            colormap[i,:] = twotone(i/(nlevels-1), scale,hue,sat)
        elif color_scheme == COLOR_WHITE:
            colormap[i,:] = np.ones((1,3))

# assign colormap to correct field
    if type == 'Field':
        colormap_field = colormap
    elif type == 'Iso':
        colormap_iso = colormap
    elif type == 'Vector':
        colormap_vect = colormap



# creates colors for every vertex from the color map
def colormaptovalues(valuefield):
    colorfield = np.zeros((50, 50, 3))
    nlevels = parameter_dict['Field']['nlevels']
    clamp_min = parameter_dict['Field']['clamp_min']
    clamp_max = parameter_dict['Field']['clamp_max']
    for i in range(0, DIM):
        for j in range(0, DIM):
            val = valuefield[i,j]
            if val < clamp_min:
                val = clamp_min
            elif val > clamp_max:
                val = clamp_max
            val = (val-clamp_min)/(clamp_max-clamp_min) * (nlevels-1)
            val = int(round(val))

            colorfield[i, j, :] = colormap_field[val,:]

    c = []
    for x in range(0, DIM - 1):
        for y in range(0, DIM - 1):
            c += [colorfield[x, y, 0], colorfield[x, y, 1], colorfield[x, y, 2]]
            c += [colorfield[x, y + 1, 0], colorfield[x, y + 1, 1], colorfield[x, y + 1, 2]]
            c += [colorfield[x + 1, y + 1, 0], colorfield[x + 1, y + 1, 1], colorfield[x + 1, y + 1, 2]]
            c += [colorfield[x, y, 0], colorfield[x, y, 1], colorfield[x, y, 2]]
            c += [colorfield[x + 1, y + 1, 0], colorfield[x + 1, y + 1, 1], colorfield[x + 1, y + 1, 2]]
            c += [colorfield[x + 1, y, 0], colorfield[x + 1, y, 1], colorfield[x + 1, y, 2]]
    return c

# make sure that black is printed when field is not shown
def blackcolors():
    return [0] * 43218


# show the colors, based on which information we want to show
def vis_color():
    valuefield = np.zeros((50, 50))
    colormap_type = parameter_dict['Field']['datatype']
    # Density
    if colormap_type == 0:
        valuefield = sim.field[-1, :, :]

    # Direction and Magnitude
    elif colormap_type == 1:
        valuefield = scale_velo_map * np.sqrt(
            sim.field[0, :, :] * sim.field[0, :, :] + sim.field[1, :, :] * sim.field[1, :, :])

    # Forces
    elif colormap_type == 2:
        valuefield = np.sqrt(sim.forces[0, :, :] * sim.forces[0, :, :] + sim.forces[1, :, :] * sim.forces[1, :, :])

    # Divergence velocity
    elif colormap_type == 3:

        valuefield = 50 * sim.divfield[:, :]

    # Divergence forces
    elif colormap_type == 4:
        valuefield = 50 * sim.divforces[:, :]

    global colors
    colors = colormaptovalues(valuefield)


# returns the vertices needed for printing the color field
def makevertices():
    threedim = parameter_dict['Field']['3d']
    hf = parameter_dict['Field']['heightfactor']
    hs = parameter_dict['Field']['heightscale']
    datatype = parameter_dict['Field']['datatype']

# get the values for printing the height plot
    if datatype == 0:
        vectfield = sim.field[-1,:,:]
    elif datatype == 1:
        vectfield = 500* np.sqrt(sim.field[0, :, :] * sim.field[0, :, :] + sim.field[1, :, :] * sim.field[1, :, :])
    elif datatype == 2:
        vectfield = 50 * np.sqrt(sim.forces[0, :, :] * sim.forces[0, :, :] + sim.forces[1, :, :] * sim.forces[1, :, :])
    elif datatype == 3:
        vectfield = 500 * sim.divfield[:, :]
    elif datatype == 4:
        vectfield = 500 * sim.divforces[:, :]

# create the vertices, with z for the heights
    v = []
    for i in range(49):
        for j in range(49):
            if threedim:
                z =  [vectfield[i,j],vectfield[i,j+1],vectfield[i+1,j+1],vectfield[i+1,j]]
                for k in range(4):
                    if z[k] < 0:
                        z[k] = -1*((-1*hf*z[k])**hs)
                    else:
                        z[k] = (hf*z[k])**hs
            else:
                z = [0,0,0,0]
            p0 = [ i / (49 / 2) - 1,j / (49 / 1.8) - 0.8, z[0]]
            p1 = [i / (49 / 2) - 1, (j + 1) / (49 / 1.8) - 0.8, z[1]]
            p2 = [(i + 1) / (49 / 2) - 1, (j + 1) / (49 / 1.8) - 0.8, z[2]]
            p3 = [(i + 1) / (49 / 2) - 1, j / (49 / 1.8) - 0.8,z[3]]
            v += p0 + p1 + p2 + p0 + p2 + p3
    return v


# draw text for the legend numbers
def drawText(input, num, rightalign=False):
    font = pygame.font.Font (None, 24)
    textSurface = font.render(str(input), True, (255,255,255,255), (0,0,0,255))
    h = 2*textSurface.get_height()/winHeight
    if rightalign:
        w = 2*textSurface.get_width()/winWidth

        position = [1-w,-1+num*h,0]
    else:
        position = [-1, -1+num*h, 0]

    textData = pygame.image.tostring(textSurface, "RGBA", True)
    glRasterPos3d(*position)
    glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)


# create vertices for legend
def makelegend():
    vertices_leg = []
    colors_leg = []
    threedim = parameter_dict['Field']['3d']
    if threedim:
        y = [-0.4,-0.4,-0.4,-0.4]
        z = [1.0,0.933,0.867,0.8]
    else:
        y = [-0.8,-0.867,-0.933,-1.0]
        z = [0,0,0,0]
    lengthf = parameter_dict['Field']['nlevels']
    lengthv = parameter_dict['Vector']['nlevels']
    lengthi = parameter_dict['Iso']['nlevels']
    length = lengthf+lengthv+lengthi

    for i in range(lengthf):
        p0 = [0.8*(i / (lengthf / 2) - 1)       , y[1], z[1]]
        p1 = [0.8*((i + 1) / (lengthf / 2) - 1) , y[1], z[1]]
        p2 = [0.8*((i + 1) / (lengthf / 2) - 1) , y[0], z[0]]
        p3 = [0.8*(i / (lengthf / 2) - 1)       , y[0], z[0]]
        vertices_leg += p0 + p1 + p2 + p0 + p2 + p3
        colval = colormap_field[i]
        colval = colval.tolist()
        colors_leg += colval * 6
    for j in range(lengthv):
        p0 = [0.8*(j / (lengthv / 2) - 1)       , y[2], z[2]]
        p1 = [0.8*((j + 1) / (lengthv / 2) - 1) , y[2], z[2]]
        p2 = [0.8*((j + 1) / (lengthv / 2) - 1) , y[1], z[1]]
        p3 = [0.8*(j / (lengthv / 2) - 1)       , y[1], z[1]]
        vertices_leg += p0 + p1 + p2 + p0 + p2 + p3
        colval = colormap_vect[j]
        colval = colval.tolist()
        colors_leg += colval * 6
    for k in range(lengthi):
        p0 = [0.8*(k / (lengthi / 2) - 1)       , y[3], z[3]]
        p1 = [0.8*((k + 1) / (lengthi / 2) - 1) , y[3], z[3]]
        p2 = [0.8*((k + 1) / (lengthi / 2) - 1) , y[2], z[2]]
        p3 = [0.8*(k / (lengthi / 2) - 1)       , y[2], z[2]]
        vertices_leg += p0 + p1 + p2 + p0 + p2 + p3
        colval = colormap_iso[k]
        colval = colval.tolist()
        colors_leg += colval * 6
    vertices_leg = np.array(vertices_leg)
    vertices_leg = vertices_leg.tolist()

#  show the legends

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
    if threedim:
        glBegin(GL_LINES)
        glVertex3f(-1,-0.8,0)
        glVertex3f(1,-0.8,0)
        glVertex3f(-1,-0.8,0)
        glVertex3f(-1,1,0)
        glVertex3f(-1,1,0)
        glVertex3f(1,1,0)
        glVertex3f(1,1,0)
        glVertex3f(1,-0.8,0)
        glEnd()

# include text
    drawText(parameter_dict['Field']['clamp_min'],2)
    drawText(parameter_dict['Vector']['clamp_min'],1)
    drawText(parameter_dict['Iso']['clamp_min'],0)
    drawText(parameter_dict['Field']['clamp_max'],2, rightalign=True)
    drawText(parameter_dict['Vector']['clamp_max'],1, rightalign=True)
    drawText(parameter_dict['Iso']['clamp_max'],0, rightalign=True)

########## VECTOR COLORING
# direction_to_color: Set the current color by mapping a direction vector (x,y)
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


# returns color depending on magnitude of vector.
def magnitude_to_color(x, y):
    RGB = np.ones(3)
    mag = np.sqrt(x * x + y * y)

    mag = 20 * mag
    clamp_min = parameter_dict['Vector']['clamp_min']
    clamp_max = parameter_dict['Vector']['clamp_max']
    nlevels = parameter_dict['Vector']['nlevels']
    if mag > clamp_max:
        mag = clamp_max
    if mag < clamp_min:
        mag = clamp_min
    RGB = colormap_vect[int(round(mag/(clamp_max-clamp_min)*(nlevels-1)))]
    return RGB


# functions that draws cones as glyphs
def drawGlyph(x, y, vx, vy, size, color, max_length):
    max_length*=1.5
    vx = np.minimum(vx, float(max_length / DIM))
    vy = np.minimum(vy, float(max_length / DIM))
    vx = np.maximum(vx, -float(max_length / DIM))
    vy = np.maximum(vy, -float(max_length / DIM))

    glBegin(GL_TRIANGLES)

    # phaux shading implemented with darker (*0.7, *0.4) and lighter corners (*1.2)
    glColor3f(color[0]*0.7, color[1]*0.7, color[2]*0.7)

    glVertex3f(x + vx, y + vy, 0.05)
    glVertex3f(x - 10 / DIM * vy, y + 10 / DIM * vx,0.05)

    glColor3f(color[0]*1.2, color[1]*1.2, color[2]*1.2)
    glVertex3f(x, y,0.06)


    glColor3f(color[0]*0.4, color[1]*0.4, color[2]*0.4)
    glVertex3f(x + vx, y + vy, 0.05)
    glVertex3f(x + 10 / DIM * vy, y - 10 / DIM * vx,0.05)

    glColor3f(color[0], color[1], color[2])
    glVertex3f(x, y,0.06)
    glEnd()


# function that draws arrows as glyphs
def drawArrow(x, y, vx, vy, size, color, max_length):
    size = max_length
    max_length *= 1.5
    vx = np.minimum(vx, float(max_length / DIM))
    vy = np.minimum(vy, float(max_length / DIM))
    vx = np.maximum(vx, -float(max_length / DIM))
    vy = np.maximum(vy, -float(max_length / DIM))

    glBegin(GL_LINES)
    glColor3f(color[0], color[1], color[2])
    glVertex2f(x + vx, y + vy)
    glVertex2f(x, y)
    glEnd()
    glBegin(GL_TRIANGLES)
# phaux shading implemented with darker (*0.7, *0.4) and lighter corners (*1.2)
    glColor3f(color[0]*0.7, color[1]*0.7, color[2]*0.7)
    glVertex3f(x + vx, y + vy,0.05)
    glVertex3f((x + 0.5 * vx) - 2 * (size / DIM) * vy, (y + 0.5 * vy) + 2 * (size / DIM) * vx,0.05)

    glColor3f(color[0]*1.2, color[1]*1.2, color[2]* 1.2)
    glVertex3f((x + 0.5 * vx), (y + 0.5 * vy),0.05)
    glVertex3f((x + 0.5 * vx), (y + 0.5 * vy),0.05)
    glColor3f(color[0]*0.4, color[1]*0.4, color[2]*0.4)
    glVertex3f(x + vx, y + vy,0.05)
    glVertex3f((x + 0.5 * vx) + 2 * (size / DIM) * vy, (y + 0.5 * vy) - 2 * (size / DIM) * vx,0.05)
    glEnd()


##### USER INPUT


# gets the drag movement of the mouse and changes the simulation values
#       according to these movements
def drag(mx, my):
    my = my
    # lmx holds the last recorded mouse position
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
# clamp the values
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


# change the displacement array. This array remembers the displacement of the
# vectors (both in x and y direction). If we don't want random seeding, the
# displacement array holds only zeros.
def displace():
    global displacement
    if parameter_dict['Vector']['displacement']:

        displacement = np.random.rand(2,50,50) -0.5

    else:
        displacement = np.zeros((2,50,50))

# perform action that was retrieved from GUI
def performAction(message):
    global parameter_dict
    global frozen


    a = message.split(':')
    action = a[0]

    if action == Action.DT_DOWN.name:
        sim.dt -= 0.001
    elif action == Action.DT_UP.name:
        sim.dt += 0.001
    elif action == Action.SET_DT.name:
        sim.dt = float(a[1])

    elif action == Action.MAG_DIR.name:
        parameter_dict['Vector']['col_mag'] += 1
        if parameter_dict['Vector']['col_mag'] > 3:
            parameter_dict['Vector']['col_mag'] = 0
    elif action == Action.VEC_SCALE_UP.name:
        parameter_dict['Vector']['vec_scale'] *= 1.2
    elif action == Action.VEC_SCALE_DOWN.name:
        parameter_dict['Vector']['vec_scale'] *= 0.8
    elif action == Action.VISC_UP.name:
        sim.visc *= 5
    elif action == Action.VISC_DOWN.name:
        sim.visc *= 0.2
    elif action == Action.COLOR_DIR.name:
        parameter_dict['Vector']['col_mag'] = parameter_dict['Vector']['col_mag'] + 1
        if parameter_dict['Vector']['col_mag'] > 3:
            parameter_dict['Vector']['col_mag'] = 1
        change_colormap('Vector')
    elif action == Action.DRAW_SMOKE.name:
        parameter_dict['Field']['show'] = not parameter_dict['Field']['show']
    elif action == Action.DRAW_VECS.name:
        parameter_dict['Vector']['show'] = not parameter_dict['Vector']['show']
    elif action == Action.DRAW_ISO.name:
        parameter_dict['Iso']['show'] = not parameter_dict['Iso']['show']
    elif action == Action.GLYPH_CHANGE.name:
        parameter_dict['Vector']['draw_glyphs'] += 1
        if parameter_dict['Vector']['draw_glyphs'] > 5:
            parameter_dict['Vector']['draw_glyphs'] = 1
    elif action == Action.SCALAR_COLOR_CHANGE.name:
        parameter_dict['Field']['color_scheme'] += 1
        if parameter_dict['Field']['color_scheme'] > COLOR_TWOTONE:
            parameter_dict['Field']['color_scheme'] = COLOR_BLACKWHITE
    elif action == Action.SCALAR_COLOR_TWOTONE.name:
        parameter_dict['Field']['color_scheme'] = COLOR_TWOTONE
        change_colormap('Field')
    elif action == Action.SCALAR_COLOR_RAINBOW.name:
        parameter_dict['Field']['color_scheme'] = COLOR_RAINBOW
        change_colormap('Field')
    elif action == Action.SCALAR_COLOR_BLACK.name:
        parameter_dict['Field']['color_scheme'] = COLOR_BLACKWHITE
        change_colormap('Field')
    elif action == Action.COLOR_MAG_BLACK.name:
        parameter_dict['Vector']['color_scheme'] = COLOR_BLACKWHITE
        change_colormap('Vector')
    elif action == Action.COLOR_MAG_RAINBOW.name:
        parameter_dict['Vector']['color_scheme'] = COLOR_RAINBOW
        change_colormap('Vector')
    elif action == Action.COLOR_MAG_TWOTONE.name:
        parameter_dict['Vector']['color_scheme'] = COLOR_TWOTONE
        change_colormap('Vector')
    elif action == Action.COLOR_MAG_WHITE.name:
        parameter_dict['Vector']['color_scheme'] = COLOR_WHITE
        change_colormap('Vector')
    elif action == Action.COLORMAP_CHANGE.name:
        parameter_dict['Field']['datatype'] += 1
        if parameter_dict['Field']['datatype'] > 4:
            parameter_dict['Field']['datatype'] = 0
    elif action == Action.FREEZE.name:
        frozen = not frozen
    elif action == Action.SET_NLEVELS_FIELD.name:
        parameter_dict['Field']['nlevels'] = int(a[1])
        change_colormap('Field')
    elif action == Action.SET_NLEVELS_ISO.name:
        parameter_dict['Iso']['nlevels'] = int(a[1])
        change_colormap('Iso')
    elif action == Action.SET_NLEVELS_VECTOR.name:
        parameter_dict['Vector']['nlevels'] = int(a[1])
        change_colormap('Vector')
    elif action == Action.GLYPH_CHANGE_N.name:
        parameter_dict['Vector']['n_glyphs'] += 5
        if parameter_dict['Vector']['n_glyphs'] > 50:
            parameter_dict['Vector']['n_glyphs'] = 5
    elif action == Action.SET_SCALE_FIELD.name:
        parameter_dict['Field']['scale'] = float(a[1])
        change_colormap('Field')
    elif action == Action.SET_SCALE_VECTOR.name:
        parameter_dict['Vector']['scale'] = float(a[1])
        change_colormap('Vector')
    elif action == Action.SET_SCALE_ISO.name:
        parameter_dict['Iso']['scale'] = float(a[1])
        change_colormap('Iso')
    elif action == Action.CHANGE_ISO_COL.name:
        parameter_dict['Iso']['color_scheme'] += 1
        if parameter_dict['Iso']['color_scheme'] > 4:
            parameter_dict['Iso']['color_scheme'] = 0
        change_colormap('Iso')
    elif action == Action.SET_ISO_MIN.name:
        parameter_dict['Iso']['iso_min'] = float(a[1])
    elif action == Action.SET_ISO_MAX.name:
        parameter_dict['Iso']['iso_max'] = float(a[1])
    elif action == Action.SET_ISO_N.name:
        parameter_dict['Iso']['iso_n'] = int(a[1])
    elif action == Action.COLOR_ISO_BLACK.name:
        parameter_dict['Iso']['color_scheme'] = 0
        change_colormap('Iso')
    elif action == Action.COLOR_ISO_RAINBOW.name:
        parameter_dict['Iso']['color_scheme'] = 1
        change_colormap('Iso')
    elif action == Action.COLOR_ISO_TWOTONE.name:
        parameter_dict['Iso']['color_scheme'] = 2
        change_colormap('Iso')
    elif action == Action.COLOR_ISO_WHITE.name:
        parameter_dict['Iso']['color_scheme'] = 3
        change_colormap('Iso')
    elif action == Action.COLORMAP_TYPE_DENSITY.name:
        parameter_dict['Field']['datatype'] = 0
    elif action == Action.COLORMAP_TYPE_VELOCITY.name:
        parameter_dict['Field']['datatype'] = 1
    elif action == Action.COLORMAP_TYPE_FORCES.name:
        parameter_dict['Field']['datatype'] = 2
    elif action == Action.COLORMAP_TYPE_DIVERGENCE.name:
        parameter_dict['Field']['datatype'] = 3
    elif action == Action.COLORMAP_TYPE_DIVERGENCE_FORCE.name:
        parameter_dict['Field']['datatype'] = 4
    elif action == Action.THREEDIM_ON_OFF.name:
        parameter_dict['Field']['3d'] = not parameter_dict['Field']['3d']
        vertices = makevertices()
    elif action == Action.HEIGHTFACTOR.name:
        parameter_dict['Field']['heightfactor'] = float(a[1])
    elif action == Action.HEIGHTSCALE.name:
        parameter_dict['Field']['heightscale'] = float(a[1])
    elif action == Action.SET_ISO_CLAMP_MIN.name:
        parameter_dict['Iso']['clamp_min'] = float(a[1])
        change_colormap('Iso')
    elif action == Action.SET_ISO_CLAMP_MAX.name:
        parameter_dict['Iso']['clamp_max'] = float(a[1])
        change_colormap('Iso')
    elif action == Action.SET_FIELD_CLAMP_MIN.name:
        parameter_dict['Field']['clamp_min'] = float(a[1])
        change_colormap('Field')
    elif action == Action.SET_FIELD_CLAMP_MAX.name:
        parameter_dict['Field']['clamp_max'] = float(a[1])
        change_colormap('Field')
    elif action == Action.SET_VECT_CLAMP_MIN.name:
        parameter_dict['Vector']['clamp_min'] = float(a[1])
        change_colormap('Vector')
    elif action == Action.SET_VECT_CLAMP_MAX.name:
        parameter_dict['Vector']['clamp_max'] = float(a[1])
        change_colormap('Vector')
    elif action == Action.SET_STREAMLINE_LENGTH.name:
        parameter_dict['Vector']['slinelength'] = int(a[1])
    elif action == Action.CHANGE_HUE_FIELD.name:
        parameter_dict['Field']['hue'] = float(a[1])
        change_colormap('Field')
    elif action == Action.CHANGE_HUE_ISO.name:
        parameter_dict['Iso']['hue'] = float(a[1])
        change_colormap('Iso')
    elif action == Action.CHANGE_HUE_VECT.name:
        parameter_dict['Vector']['hue'] = float(a[1])
        change_colormap('Vector')
    elif action == Action.CHANGE_SAT_FIELD.name:
        parameter_dict['Field']['sat'] = float(a[1])
        change_colormap('Field')
    elif action == Action.CHANGE_SAT_ISO.name:
        parameter_dict['Iso']['sat'] = float(a[1])
        change_colormap('Iso')
    elif action == Action.CHANGE_SAT_VECT.name:
        parameter_dict['Vector']['sat'] = float(a[1])
        change_colormap('Vector')
    elif action == Action.VELO_TO_FORCE.name:
        parameter_dict['Vector']['velocity'] = not parameter_dict['Vector']['velocity']
    elif action == Action.DISPLACE.name:
        parameter_dict['Vector']['displacement'] = not parameter_dict['Vector']['displacement']
        displace()



# function that gets the keyboard input, which is used for controlling parameters
#       of the simulation.
def keyboard(key):
    global q
    if key == pygame.K_f:
        q.put(Action.FREEZE.name)



# retrieve information from the GUI
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

# interpolate velocity for streamlines and streamtubes.
def interpolateVelocity(x, y, vectfield):
    x_floor = int(np.floor(x))
    x_ceil  = int(np.ceil(x))
    y_floor = int(np.floor(y))
    y_ceil  = int(np.ceil(y))

    x = x-x_floor
    y = y-y_floor

# velocity at point x,y in direction x (=vx) and direction y (=vy)
    vx = vectfield[0, x_floor, y_floor] * (1 - x) * (1 - y) + \
         vectfield[0, x_ceil,  y_floor] * x       * (1 - y) + \
         vectfield[0, x_floor, y_ceil ] * (1 - x) * y       + \
         vectfield[0, x_ceil,  y_ceil ] * x * y

    vy = vectfield[1, x_floor, y_floor] * (1 - x) * (1 - y) + \
         vectfield[1, x_ceil, y_floor] * x * (1 - y) + \
         vectfield[1, x_floor, y_ceil] * (1 - x) * y + \
         vectfield[1, x_ceil, y_ceil] * x * y

    return [vx, vy]

def main():

    print("Fluid Flow Simulation and Visualization\n")
    print("=======================================\n")
    print("Click and drag the mouse to steer the flow!\n")
    print("f:   freeze the screen\n")


    clock = pygame.time.Clock()

    thread = Thread(target=getGuiInput)
    thread.start()

# initialize colormaps
    change_colormap('Field')
    change_colormap('Iso')
    change_colormap('Vector')

# initialize the vertices to print
    vertices = makevertices()
    global colors
    colors = colormaptovalues(sim.field[-1, :, :])

# make a buffer for the vertices and colors
    vertices_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vertices_vbo)
    glBufferData(GL_ARRAY_BUFFER, len(vertices) * 4, (c_float * len(vertices))(*vertices), GL_STATIC_DRAW)

    colors_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, colors_vbo)
    glBufferData(GL_ARRAY_BUFFER, len(colors) * 4, (c_float * len(colors))(*colors), GL_STATIC_DRAW)


    running = True
    while running:

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

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragbool = False
            if event.type == pygame.KEYDOWN:
                keyboard(event.key)
        # if the mouse button is pressed down, influence the simulation
        if dragbool:
            try:
                mx, my = event.pos
                drag(mx, my)
            except AttributeError:
                pass
        # if there is an element in the queue, perform it
        while not q.empty():
            performAction(q.get())

# do a simulation step
        sim.do_one_simulation_step(frozen)


        threedim = parameter_dict['Field']['3d']
# change the vertices in z direction if a heightplot is asked
        if threedim:
            vertices = makevertices()
# change colors
        vis_color()

        glBindBuffer(GL_ARRAY_BUFFER, vertices_vbo)
        glBufferData(GL_ARRAY_BUFFER, len(vertices) * 4, (c_float * len(vertices))(*vertices), GL_STATIC_DRAW)

        glEnableClientState(GL_COLOR_ARRAY)

        glBindBuffer(GL_ARRAY_BUFFER, vertices_vbo)
        glVertexPointer(3, GL_FLOAT, 0, None)
        if not parameter_dict['Field']['show']:
            colors = blackcolors()
        glBindBuffer(GL_ARRAY_BUFFER, colors_vbo)
        glBufferData(GL_ARRAY_BUFFER, len(colors) * 4, (c_float * len(colors))(*colors), GL_STATIC_DRAW)
        glColorPointer(3, GL_FLOAT, 0, None)

        glDrawArrays(GL_TRIANGLES, 0, 43218)


        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
# change perspective if we look in 3d
        if threedim:
            gluPerspective(90, 1, 0.1, 10)
            gluLookAt(0,-1.5, 0.4,0, 0, 0, 0, -0.5, 0.7)
            glDepthFunc(GL_LESS)
        else:
            gluLookAt(0,0, 1,0, 0, 0, 0, 1, 0)
            glDepthFunc(GL_ALWAYS)
        glClearColor(0.0, 0.0, 0.0, 1.0)
# retrieve parameters from dict
        draw_glyphs = parameter_dict['Vector']['draw_glyphs']
        n_glyphs = parameter_dict['Vector']['n_glyphs']
        vec_scale = parameter_dict['Vector']['vec_scale']
        show_vecs = parameter_dict['Vector']['show']
        global displacement

# show vectors
        if show_vecs:
# retrieve vector field to show
            if parameter_dict['Vector']['velocity']:
                vectfield = sim.field[0:2,:,:]
            else:
                vectfield = sim.forces

# create list with all the vectors before we print
            glNewList(1, GL_COMPILE)
            step = DIM / n_glyphs
            for i in range(0, n_glyphs):
                for j in range(0, n_glyphs):
                    # print hedgehogs
                    if draw_glyphs == 1 :
                        glBegin(GL_LINES)
                        x = round(i * step)
                        y = round(j * step)
                        color = np.ones(3)
                        # which data to encode in the vector color
                        if parameter_dict['Vector']['col_mag']==1:
                            color = magnitude_to_color(vectfield[0, x, y], vectfield[1, x, y])
                        elif parameter_dict['Vector']['col_mag']==2:
                            color = direction_to_color(vectfield[0, x,y], vectfield[1, x, y])
                        elif parameter_dict['Vector']['col_mag']==3:
                            xx = 49 - x
                            yy = 49 - y
                            if not (xx+yy) > 96:
                                color = colors[18*xx + 882*yy : 18*xx + 882*yy + 3]
                            elif (xx+yy) == 98:
                                color = colors[18*xx + 882*yy -6: 18*xx + 882*yy -3]
                            elif xx == 49:
                                color = colors[18*xx + 882*yy -3 : 18*xx + 882*yy]
                            else:
                                color = colors[18*xx + 882*yy +3 : 18*xx + 882*yy +6]
                        # displace the vectors a bit
                        id = i + displacement[0,i,j]
                        jd = j + displacement[1,i,j]

                        # draw the hedgehogs
                        glColor3f(color[0], color[1], color[2])

                        glVertex2f((((id + 0.5) * step / (49 / 2)) - 1), (((jd + 0.5) * step / (49 / 1.8)) - 0.8))
                        glVertex2f((((id + 0.5) * step / (49 / 2)) - 1) + vec_scale * vectfield[0, x, y],
                                   (((jd + 0.5) * step / (49 / 1.8)) - 0.8) + vec_scale * vectfield[1, x, y])
                        glEnd()
# other vector types, similar to hedgehogs
                    if draw_glyphs >= 2:

                        x = i * step
                        y = j * step
                        id = i + displacement[0,i,j]
                        jd = j + displacement[1,i,j]
                        vx = step * vectfield[0, round(x), round(y)]
                        vy = step * vectfield[1, round(x), round(y)]
                        x2 = (id + 0.5) * step / ((DIM - 1) / 2) - 1
                        y2 = (jd + 0.5) * step / ((DIM - 1) / 1.8) - 0.8
                        x = round(x)
                        y = round(y)
                        color = np.ones(3)
                        if parameter_dict['Vector']['col_mag'] == 1:
                            color = magnitude_to_color(vectfield[0, round(x), round(y)], vectfield[1, round(x), round(y)])
                        elif parameter_dict['Vector']['col_mag'] == 2:
                            color = direction_to_color(vectfield[0, round(x), round(y)], vectfield[1, round(x), round(y)])
                        elif parameter_dict['Vector']['col_mag']==3:
                            xx = y
                            yy = x
                            if yy < 49 & xx < 49:
                                color = colors[18*xx + 882*yy : 18*xx + 882*yy + 3]

                            elif (xx+yy) == 98:

                                color = colors[18*(xx-1) + 882*(yy-1) -6: 18*(xx-1) + 882*(yy-1) -3]
                            elif xx == 49:
                                color = colors[18*(xx-1) + 882*(yy-1) -3 : 18*(xx-1) + 882*(yy-1)]
                            else:
                                color = colors[18*(xx-1) + 882*(yy-1) +3 : 18*(xx-1) + 882*(yy-1) +6]
                        size = sim.field[-1, round(x), round(y)]
                        # draw either glyphs or arrows
                        if draw_glyphs == 2:
                            drawGlyph(x2, y2, vx, vy, size, color, step)
                        elif draw_glyphs == 3:
                            drawArrow(x2, y2, vx, vy, size, color, step)
                        # draw streamlines
                        elif draw_glyphs == 4:
                            glBegin(GL_LINES)
                            T = parameter_dict['Vector']['slinelength']
                            x_d = x2
                            y_d = y2
                            # draw T times a part of the streamline
                            for t in range(T):
                                if x > 49 or y > 49 or x < 0 or y < 0:
                                    break

                                # Determine the velocity on (x, y) by means of interpolation
                                [vx, vy] = interpolateVelocity(x, y, vectfield)

                                # Determine the velocity magnitude
                                v_l = np.sqrt(vx*vx + vy*vy)

                                # Stop if the velocity is too low
                                if v_l < 0.001:
                                    break

                                # Determine (x, y) at new time t
                                x_t = x + vx / (v_l)
                                y_t = y + vy / (v_l)
                                color = np.ones(3)

                                # Color the streamline
                                if parameter_dict['Vector']['col_mag'] == 1:
                                    # Magnitude to color
                                    color = magnitude_to_color(vx, vy)
                                elif parameter_dict['Vector']['col_mag'] == 2:
                                    # Direction to color
                                    color = direction_to_color(vx, vy)
                                elif parameter_dict['Vector']['col_mag'] == 3:
                                    # Scalar field value to color
                                    xx = y
                                    yy = x
                                    if yy < 49 and xx < 49:
                                        color = colors[18*xx + 882*yy : 18*xx + 882*yy + 3]

                                    elif (xx+yy) == 98:

                                        color = colors[18*(xx-1) + 882*(yy-1) -6: 18*(xx-1) + 882*(yy-1) -3]
                                    elif xx == 49:
                                        color = colors[18*(xx-1) + 882*(yy-1) -3 : 18*(xx-1) + 882*(yy-1)]
                                    else:
                                        color = colors[18*(xx-1) + 882*(yy-1) +3 : 18*(xx-1) + 882*(yy-1) +6]
                                glColor3f(color[0], color[1], color[2])

                                # Draw the line segment
                                glVertex3f(x_d, y_d,0.05)
                                x_d += vx/(v_l*49)
                                y_d += vy/(v_l*49)
                                glVertex3f(x_d, y_d,0.05)
                                x = x_t
                                y = y_t
                            glEnd()
                        else:
                            # draw stream tubes
                            glBegin(GL_TRIANGLES)
                            T = parameter_dict['Vector']['slinelength']
                            x_d = x2
                            y_d = y2
                            for t in range(T):
                                if x > 49 or y > 49 or x < 0 or y < 0:
                                    break
                                [vx, vy] = interpolateVelocity(x, y,vectfield)
                                v_l = np.sqrt(vx * vx + vy * vy)

                                x_t = x + vx / (v_l)
                                y_t = y + vy / (v_l)
                                color = np.ones(3)
                                if parameter_dict['Vector']['col_mag'] == 1:
                                    color = magnitude_to_color(vx, vy)
                                elif parameter_dict['Vector']['col_mag'] == 2:
                                    color = direction_to_color(vx, vy)
                                elif parameter_dict['Vector']['col_mag'] == 3:
                                    xx = y
                                    yy = x
                                    if yy < 49 and xx < 49:
                                        color = colors[18*xx + 882*yy : 18*xx + 882*yy + 3]

                                    elif (xx+yy) == 98:
                                        color = colors[18*(xx-1) + 882*(yy-1) -6: 18*(xx-1) + 882*(yy-1) -3]
                                    elif xx == 49:
                                        color = colors[18*(xx-1) + 882*(yy-1) -3 : 18*(xx-1) + 882*(yy-1)]
                                    else:
                                        color = colors[18*(xx-1) + 882*(yy-1) +3 : 18*(xx-1) + 882*(yy-1) +6]

                                # Make the triangle size depend on t
                                size = 70 - 70*t/T

                                # Remember corner coordinates for 'supporting' triangles
                                lx = x_d - (size / DIM) * vy/(v_l * 49)
                                ly = y_d + (size / DIM) * vx/(v_l * 49)
                                rx = x_d + (size / DIM) * vy / (v_l * 49)
                                ry = y_d - (size / DIM) * vx / (v_l * 49)

                                # Draw 'supporting' triangles to approach cone shape
                                if t > 0:
                                    glColor4f(color[0], color[1], color[2], 1 - (t-1) / T)
                                    glVertex3f(bottom_lx, bottom_ly,0.05)
                                    glColor4f(color[0], color[1], color[2], 1 - t / T)
                                    glVertex3f(lx, ly,0.05)
                                    glVertex3f(x_d, y_d,0.05)

                                    glColor4f(color[0], color[1], color[2], 1 - (t-1) / T)
                                    glVertex3f(bottom_rx, bottom_ry,0.05)
                                    glColor4f(color[0], color[1], color[2], 1 - t / T)
                                    glVertex3f(rx, ry,0.05)
                                    glVertex3f(x_d, y_d,0.05)

                                # Make opacity depend on t
                                glColor4f(color[0], color[1], color[2], 1 - t / T)
                                glVertex3f(lx, ly,0.05)
                                glVertex3f(rx, ry,0.05)

                                x_d += vx / (v_l * 49)
                                y_d += vy / (v_l * 49)
                                glColor4f(color[0], color[1], color[2], 1 - (t + 1) / T)
                                glVertex3f(x_d, y_d,0.05)

                                bottom_lx = lx
                                bottom_ly = ly
                                bottom_rx = rx
                                bottom_ry = ry

                                x = x_t
                                y = y_t
                                if math.isnan(x):
                                    break
                                elif math.isnan(y):
                                    break
                            glEnd()
            glEndList()

        # show isolines if necessary
        if parameter_dict['Iso']['show']:
            isolines()

        # draw vectors from list if necessary
        if show_vecs:
            glCallList(1)

        # refresh legend
        makelegend()
        pygame.display.flip()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

# initialize visualization
pygame.init()
pygame.font.init()

screen = pygame.display.set_mode((winWidth, winHeight + 55), pygame.OPENGL | pygame.DOUBLEBUF, 24)

glEnable(GL_DEPTH_TEST)
glEnable(GL_BLEND);
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
glEnableClientState(GL_VERTEX_ARRAY)
main()
