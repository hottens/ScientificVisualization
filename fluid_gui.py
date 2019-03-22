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

def option_changed_fielddata(*args):
    callBack(ftype_dict[field_datatype_dropdown.get()])

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
ftype_dict = {'Density': Action.COLORMAP_TYPE_DENSITY,
                      'Divergence': Action.COLORMAP_TYPE_DIVERGENCE,
                      'Velocity': Action.COLORMAP_TYPE_VELOCITY,
                      'Forces': Action.COLORMAP_TYPE_FORCES
                      }
FShow = tkinter.Button(f1, text = "Show Field",command=lambda: callBack(Action.DRAW_SMOKE))


FScale = tkinter.Scale(f1, from_=0.00, to=2.00, resolution=0.01, orient='horizontal')
FScale.set(1.00)
FScaleB = tkinter.Button(f1, text="Set Scale", command=lambda: callBack(Action.SET_SCALE_FIELD, FScale.get()))

FNlevels = tkinter.Scale(f1, from_=2.0, to=256.0, orient='horizontal')
FNlevels.set(50)
FNlevelsB = tkinter.Button(f1, text="Set Number of Colors", command=lambda: callBack(Action.SET_NLEVELS_FIELD, [FNlevels.get()]))

f_data = ttk.Labelframe(f1, text='Data in Field')
field_datatype_dropdown = tkinter.StringVar()
field_datatype_dropdown.set('Density')
field_datatype_dropdown.trace('w', option_changed_fielddata)
field_data_dropdown = tkinter.OptionMenu(f_data, field_datatype_dropdown, *ftype_dict)



### Vector
VShow = tkinter.Button(f2, text = "Show Vectors",command=lambda: callBack(Action.DRAW_VECS))

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
B = tkinter.Button(v_type, text="Change number of vectors", command=lambda: callBack(Action.GLYPH_CHANGE_N))

VScale = tkinter.Scale(f2, from_=0.00, to=2.00, resolution=0.01, orient='horizontal')
VScale.set(1.00)
VScaleB = tkinter.Button(f2, text="Set Scale", command=lambda: callBack(Action.SET_SCALE_VECTOR, VFScale.get()))

VNlevels = tkinter.Scale(f2, from_=2, to=256, orient='horizontal')
VNlevels.set(50)
VNlevelsB = tkinter.Button(f2, text="Set Number of Colors", command=lambda: callBack(Action.SET_NLEVELS_VECTOR, VNlevels.get()))




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

IScale = tkinter.Scale(f3, from_=0.00, to=2.00, resolution=0.01, orient='horizontal')
IScale.set(1.00)
IScaleB = tkinter.Button(f3, text="Set Scale", command=lambda: callBack(Action.SET_SCALE_ISO, IScale.get()))

INlevels = tkinter.Scale(f3, from_=2, to=256, orient='horizontal')
INlevels.set(50)
INlevelsB = tkinter.Button(f3, text="Set Number of Colors", command=lambda: callBack(Action.SET_ILEVELS_FIELD, INlevels.get()))

IShow = tkinter.Button(f3, text = "Show Isolines",command=lambda: callBack(Action.DRAW_ISO))


DtSlider = tkinter.Scale(top, from_=0.1, to=1.0, resolution=0.001, orient='horizontal')
DtSlider.set(0.4)
DtButton = tkinter.Button(top, text="Set dt", command=lambda: callBack(Action.SET_DT, DtSlider.get()))


nb.pack()
B.pack()
field_data_dropdown.pack()
ColorDir.pack()
FScale.pack()
FScaleB.pack()
DtSlider.pack()
DtButton.pack()
isoMinSlider.pack()
isoMinButton.pack()
isoMaxSlider.pack()
isoMaxButton.pack()
isoNSlider.pack()
isoNButton.pack()
VScale.pack()
VScaleB.pack()
IScale.pack()
IScaleB.pack()

FNlevels.pack()
FNlevelsB.pack()
VNlevels.pack()
VNlevelsB.pack()

INlevels.pack()
INlevelsB.pack()

FShow.pack()
VShow.pack()
IShow.pack()

vecColDropdown.pack()
isoColDropdown.pack()


top.mainloop()
