import pygame
from OpenGL.GL import *
from ctypes import *
import fluid2 as fl


pygame.init()
screen = pygame.display.set_mode((800, 600), pygame.OPENGL | pygame.DOUBLEBUF, 24)
glViewport(0, 0, 800, 600)
glClearColor(0.0, 0.5, 0.5, 1.0)
glEnableClientState(GL_VERTEX_ARRAY)

vertices = [-1.0, 0.0, 1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
colors = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]





vertices_vbo = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vertices_vbo)
glBufferData(GL_ARRAY_BUFFER, len(vertices) * 4, (c_float * len(vertices))(*vertices), GL_STATIC_DRAW)

colors_vbo = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, colors_vbo)
glBufferData(GL_ARRAY_BUFFER, len(colors) * 4, (c_float * len(colors))(*colors), GL_STATIC_DRAW)

running = True
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    glClear(GL_COLOR_BUFFER_BIT)

    glEnableClientState(GL_COLOR_ARRAY)

    glBindBuffer(GL_ARRAY_BUFFER, vertices_vbo)
    glVertexPointer(3, GL_FLOAT, 0, None)

    glBindBuffer(GL_ARRAY_BUFFER, colors_vbo)
    glColorPointer(3, GL_FLOAT, 0, None)


    glDrawArrays(GL_TRIANGLES, 0, 6)

    pygame.display.flip()
