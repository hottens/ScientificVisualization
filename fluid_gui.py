import socket
from fluid_actions import Action
import tkinter
import numpy as np

top = tkinter.Tk()


def option_changed(*args):
    callBack(VectorColoringDict[vector_coloring_dropdown.get()])


def callBack(action, value=None):
    print(action)
    clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clientsocket.connect(('localhost', 8089))
    message = action.name
    if value is not None:
        message += ':' + str(value)
    clientsocket.send(message.encode('utf-8'))
    print(action.name.encode('utf-8'))


B = tkinter.Button(top, text="Hello", command=lambda: callBack(Action.GLYPH_CHANGE_N))

ColorDir = tkinter.Button(top, text="Color Dir", command=lambda: callBack(Action.COLOR_DIR))

VectorColoringDict = {'Black and White': Action.COLOR_MAG_BLACK,
                      'Rainbow': Action.COLOR_MAG_RAINBOW,
                      'Twotone': Action.COLOR_MAG_TWOTONE
                      }
vector_coloring_dropdown = tkinter.StringVar()
vector_coloring_dropdown.set('Black and White')
vector_coloring_dropdown.trace('w', option_changed)
vecColDropdown = tkinter.OptionMenu(top, vector_coloring_dropdown, *VectorColoringDict)

S = tkinter.Scale(top, from_=0.00, to=2.00, resolution=0.01)
S.set(1.00)
Sb = tkinter.Button(top, text="Set Scale", command=lambda: callBack(Action.SET_SCALE, S.get()))
DtSlider = tkinter.Scale(top, from_=0.1, to=1.0, resolution=0.001)
DtSlider.set(0.4)
DtButton = tkinter.Button(top, text="Set dt", command=lambda: callBack(Action.SET_DT, DtSlider.get()))

B.pack()
ColorDir.pack()
S.pack()
Sb.pack()
DtSlider.pack()
DtButton.pack()
vecColDropdown.pack()

top.mainloop()
