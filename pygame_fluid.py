import pygame
from OpenGL.GL import *
from ctypes import *
import numpy as np

pygame.init()
screen = pygame.display.set_mode((800, 600), pygame.OPENGL | pygame.DOUBLEBUF, 24)
glViewport(0, 0, 800, 600)
glClearColor(0.0, 0.5, 0.5, 1.0)
glEnableClientState(GL_VERTEX_ARRAY)

# Create all triangle vertices
vertices = []
for i in range(50):
    for j in range(50):
        p0 = [i, j, 0]
        p1 = [i, j+1, 0]
        p2 = [i+1, j+1, 0]
        p3 = [i+1, j, 0]
        vertices += p0 + p1 + p2 + p0 + p2 + p3
vertices = np.array(vertices)
vertices = vertices/(49/2)-1
vertices = vertices.tolist()
print(vertices)
colors = vertices

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

    glDrawArrays(GL_TRIANGLES, 0, 45000)

    pygame.display.flip()
