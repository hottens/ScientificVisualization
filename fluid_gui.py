import socket
from fluid_actions import Action
import tkinter
import numpy as np

top = tkinter.Tk()


def option_changed_vc(*args):
    callBack(VectorColoringDict[vector_coloring_dropdown.get()])

def option_changed_iso(*args):
    callBack(IsoColoringDict[iso_coloring_dropdown.get()])

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

VectorColoringDict = {'Vec Col Black and White': Action.COLOR_MAG_BLACK,
                      'Vec Col Rainbow': Action.COLOR_MAG_RAINBOW,
                      'Vec Col Twotone': Action.COLOR_MAG_TWOTONE
                      }
vector_coloring_dropdown = tkinter.StringVar()
vector_coloring_dropdown.set('Vec Col Black and White')
vector_coloring_dropdown.trace('w', option_changed_vc)
vecColDropdown = tkinter.OptionMenu(top, vector_coloring_dropdown, *VectorColoringDict)

IsoColoringDict =    {'Iso Col Black and White': Action.COLOR_ISO_BLACK,
                      'Iso Col Rainbow': Action.COLOR_ISO_RAINBOW,
                      'Iso Col Twotone': Action.COLOR_ISO_TWOTONE,
                      'Iso Col White': Action.COLOR_ISO_WHITE
                      }
iso_coloring_dropdown = tkinter.StringVar()
iso_coloring_dropdown.set('Iso Col White')
iso_coloring_dropdown.trace('w', option_changed_iso)
isoColDropdown = tkinter.OptionMenu(top, iso_coloring_dropdown, *IsoColoringDict)

S = tkinter.Scale(top, from_=0.00, to=2.00, resolution=0.01)
S.set(1.00)
Sb = tkinter.Button(top, text="Set Scale", command=lambda: callBack(Action.SET_SCALE, S.get()))
DtSlider = tkinter.Scale(top, from_=0.1, to=1.0, resolution=0.001)
DtSlider.set(0.4)
DtButton = tkinter.Button(top, text="Set dt", command=lambda: callBack(Action.SET_DT, DtSlider.get()))
isoMinSlider = tkinter.Scale(top, from_=0.01, to = 4.99, resolution=0.001)
isoMinSlider.set(0.7)
isoMinButton = tkinter.Button(top, text="Set iso min", command=lambda: callBack(Action.SET_ISO_MIN, isoMinSlider.get()))
isoMaxSlider = tkinter.Scale(top, from_=0.02, to = 5.00, resolution=0.001)
isoMaxSlider.set(1.0)
isoMaxButton = tkinter.Button(top, text="Set iso max", command=lambda: callBack(Action.SET_ISO_MAX, isoMaxSlider.get()))
isoNSlider = tkinter.Scale(top, from_=1, to = 50)
isoNSlider.set(1)
isoNButton = tkinter.Button(top, text="Set iso n", command=lambda: callBack(Action.SET_ISO_N, isoNSlider.get()))


B.pack()
ColorDir.pack()
S.pack()
Sb.pack()
DtSlider.pack()
DtButton.pack()
isoMinSlider.pack()
isoMinButton.pack()
isoMaxSlider.pack()
isoMaxButton.pack()
isoNSlider.pack()
isoNButton.pack()


vecColDropdown.pack()
isoColDropdown.pack()

top.mainloop()
