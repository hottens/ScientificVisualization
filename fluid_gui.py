import socket
from fluid_actions import Action
import tkinter
from tkinter import ttk
import numpy as np

top = tkinter.Tk()
width = 200

#Make the notebook
nb = ttk.Notebook(top)

#Make 1st tab
f1 = tkinter.Frame(nb)
nb.add(f1, text="Field")
#Make 2nd tab
f2 = tkinter.Frame(nb)
nb.add(f2, text="Vectors")
#Make 3rd tab
f3 = tkinter.Frame(nb)
nb.add(f3, text="Isolines")

nb.select(f1)
nb.enable_traversal()

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
        for v in value:
            message += ':' + str(v)
    clientsocket.send(message.encode('utf-8'))
    print(action.name.encode('utf-8'))


### Field
S = tkinter.Scale(f1, from_=0.00, to=2.00, resolution=0.01, orient='horizontal')
S.set(1.00)
Sb = tkinter.Button(f1, text="Set Scale", command=lambda: callBack(Action.SET_SCALE, S.get()))


### Vector
VectorColoringDict = {'Black and White': Action.COLOR_MAG_BLACK,
                      'Rainbow': Action.COLOR_MAG_RAINBOW,
                      'Twotone': Action.COLOR_MAG_TWOTONE
                      }
v_color = ttk.Labelframe(f2, text='Color')
vector_coloring_dropdown = tkinter.StringVar()
vector_coloring_dropdown.set('Black and White')
vector_coloring_dropdown.trace('w', option_changed_vc)
vecColDropdown = tkinter.OptionMenu(v_color, vector_coloring_dropdown, *VectorColoringDict)
ColorDir = tkinter.Button(v_color, text="Color Dir", command=lambda: callBack(Action.COLOR_DIR))

v_type = ttk.Labelframe(f2, text='Type')
B = tkinter.Button(v_type, text="Change vector type", command=lambda: callBack(Action.GLYPH_CHANGE_N))


v_color.pack(side='left', fill='x')
v_type.pack(side='left', fill='x')

### Iso
IsoColoringDict =    {'Black to White': Action.COLOR_ISO_BLACK,
                      'Rainbow': Action.COLOR_ISO_RAINBOW,
                      'Twotone': Action.COLOR_ISO_TWOTONE,
                      'White': Action.COLOR_ISO_WHITE
                      }
iso_coloring_dropdown = tkinter.StringVar()
iso_coloring_dropdown.set('White')
iso_coloring_dropdown.trace('w', option_changed_iso)
isoColDropdown = tkinter.OptionMenu(f3, iso_coloring_dropdown, *IsoColoringDict)
isoMinSlider = tkinter.Scale(f3, from_=0.01, to = 4.99, resolution=0.001, orient='horizontal')
isoMinSlider.set(0.7)
isoMinButton = tkinter.Button(f3, text="Set iso min", command=lambda: callBack(Action.SET_ISO_MIN, isoMinSlider.get()))
isoMaxSlider = tkinter.Scale(f3, from_=0.02, to = 5.00, resolution=0.001, orient='horizontal')
isoMaxSlider.set(1.0)
isoMaxButton = tkinter.Button(f3, text="Set iso max", command=lambda: callBack(Action.SET_ISO_MAX, isoMaxSlider.get()))
isoNSlider = tkinter.Scale(f3, from_=1, to = 50, orient='horizontal')
isoNSlider.set(1)
isoNButton = tkinter.Button(f3, text="Set iso n", command=lambda: callBack(Action.SET_ISO_N, isoNSlider.get()))



DtSlider = tkinter.Scale(top, from_=0.1, to=1.0, resolution=0.001, orient='horizontal')
DtSlider.set(0.4)
DtButton = tkinter.Button(top, text="Set dt", command=lambda: callBack(Action.SET_DT, DtSlider.get()))


nb.pack()
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
