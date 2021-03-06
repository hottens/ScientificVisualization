import socket
from fluid_actions import Action
import tkinter
from tkinter import ttk
import numpy as np

# Main widget
top = tkinter.Tk()
width = 200

velo = True
coldir = 0
n_glyphs = 16

#Make the notebook and tabs
nb = ttk.Notebook(top)
f1 = tkinter.Frame(nb)
nb.add(f1, text="Field")
f2 = tkinter.Frame(nb)
nb.add(f2, text="Vectors")
f3 = tkinter.Frame(nb)
nb.add(f3, text="Isolines")

nb.select(f1)
nb.enable_traversal()

### Some functions to give callback to visualization application
def option_changed_vc(*args):
    callBack(VectorColoringDict[vector_coloring_dropdown.get()])

def option_changed_fielddata(*args):
    callBack(ftype_dict[field_datatype_dropdown.get()])

def option_changed_field3ddata(*args):
    callBack(f3dtype_dict[field_3ddatatype_dropdown.get()])

def option_changed_fieldcolor(*args):
    callBack(fcolor_dict[field_color_dropdown.get()])

def option_changed_iso(*args):
    callBack(IsoColoringDict[iso_coloring_dropdown.get()])

def callBack(action, value=None):
    global velo
    global coldir
    global n_glyphs
    print(action)
    clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clientsocket.connect(('localhost', 8089))
    message = action.name
    if value is not None:
        message += ':' + str(value)
    clientsocket.send(message.encode('utf-8'))
    print(action.name.encode('utf-8'))
    if action == Action.VELO_TO_FORCE:
        velo = not velo
    if action == Action.COLOR_DIR:
        coldir += 1
        if coldir > 2:
            coldir = 0
    if action == Action.GLYPH_CHANGE_N:
        n_glyphs += 5
        if n_glyphs > 50:
            n_glyphs = 5


### Field
ftype_dict = {'Density': Action.COLORMAP_TYPE_DENSITY,

              'Divergence Velocity field': Action.COLORMAP_TYPE_DIVERGENCE,
              'Divergence Force field': Action.COLORMAP_TYPE_DIVERGENCE_FORCE,
              'Velocity': Action.COLORMAP_TYPE_VELOCITY,
              'Forces': Action.COLORMAP_TYPE_FORCES
              }
f3dtype_dict = {'Height: Density': Action.THREEDIM_TYPE_DENSITY,

              'Height: Divergence Velocity field': Action.THREEDIM_TYPE_DIVERGENCE,
              'Height: Divergence Force field': Action.THREEDIM_TYPE_DIVERGENCE_FORCE,
              'Height: Velocity': Action.THREEDIM_TYPE_VELOCITY,
              'Height: Forces': Action.THREEDIM_TYPE_FORCES
              }
fcolor_dict = {'Black and White': Action.SCALAR_COLOR_BLACK,
               'Rainbow': Action.SCALAR_COLOR_RAINBOW,
               'Twotone': Action.SCALAR_COLOR_TWOTONE
              }


field_datatype_dropdown = tkinter.StringVar()
field_datatype_dropdown.set('Density')
field_datatype_dropdown.trace('w', option_changed_fielddata)
field_data_dropdown = tkinter.OptionMenu(f1, field_datatype_dropdown, *ftype_dict)

field_3ddatatype_dropdown = tkinter.StringVar()
field_3ddatatype_dropdown.set('Height: Density')
field_3ddatatype_dropdown.trace('w', option_changed_field3ddata)
field_3ddata_dropdown = tkinter.OptionMenu(f1, field_3ddatatype_dropdown, *f3dtype_dict)


field_color_dropdown = tkinter.StringVar()
field_color_dropdown.set('Black and White')
field_color_dropdown.trace('w', option_changed_fieldcolor)
field_c_dropdown = tkinter.OptionMenu(f1, field_color_dropdown, *fcolor_dict)


FShow = tkinter.Button(f1, text = "Show Field",command=lambda: callBack(Action.DRAW_SMOKE))
F3d = tkinter.Button(f1, text = "3D On/Off",command=lambda: callBack(Action.THREEDIM_ON_OFF))

FScale = tkinter.Scale(f1, from_=0.01, to=2.00, resolution=0.01, orient='horizontal')
FScale.set(1.00)
FScaleB = tkinter.Button(f1, text="Set Scale", command=lambda: callBack(Action.SET_SCALE_FIELD, FScale.get()))

FXlookS = tkinter.Scale(f1, from_=-1.5, to=1.5, resolution=0.01, orient='horizontal')
FXlookS.set(0.00)
FXlookB = tkinter.Button(f1, text="Set X viewpoint", command=lambda: callBack(Action.CHANGE_X_LOOK, FXlookS.get()))

FYlookS = tkinter.Scale(f1, from_=-0.5, to=0.5, resolution=0.01, orient='horizontal')
FYlookS.set(0.00)
FYlookB = tkinter.Button(f1, text="Zoom", command=lambda: callBack(Action.CHANGE_Y_LOOK, -1.5 + FYlookS.get()))


FNlevels = tkinter.Scale(f1, from_=2.0, to=256.0, orient='horizontal')
FNlevels.set(50)
FNlevelsB = tkinter.Button(f1, text="Set Number of Colors", command=lambda: callBack(Action.SET_NLEVELS_FIELD, FNlevels.get()))

FHeight = tkinter.Scale(f1, from_ =0.01, to= 0.1, resolution = 0.01, orient = 'horizontal')
FHeight.set(0.05)
FHeightB = tkinter.Button(f1, text = "Set height factor for 3d", command = lambda: callBack(Action.HEIGHTFACTOR, FHeight.get()))
FHeightScale = tkinter.Scale(f1, from_ =0.01, to= 2.00, resolution = 0.01, orient = 'horizontal')
FHeightScale.set(1.0)
FHeightScaleB = tkinter.Button(f1, text = "Set height scale for 3d", command = lambda: callBack(Action.HEIGHTSCALE, FHeightScale.get()))

fieldClaMinSlider = tkinter.Scale(f1, from_=-1.00, to = 4.99, resolution=0.001, orient='horizontal')
fieldClaMinSlider.set(0.0)
fieldClaMinButton = tkinter.Button(f1, text="Set field clamp min", command=lambda: callBack(Action.SET_FIELD_CLAMP_MIN, fieldClaMinSlider.get()))
fieldClaMaxSlider = tkinter.Scale(f1, from_=-0.99, to = 5.00, resolution=0.001, orient='horizontal')
fieldClaMaxSlider.set(1.0)
fieldClaMaxButton = tkinter.Button(f1, text="Set field clamp max", command=lambda: callBack(Action.SET_FIELD_CLAMP_MAX, fieldClaMaxSlider.get()))


FHue = tkinter.Scale(f1, from_ =0, to= 5, orient = 'horizontal')
FHue.set(0)
FHueB = tkinter.Button(f1, text = "Set hue", command = lambda: callBack(Action.CHANGE_HUE_FIELD, FHue.get()))


FSat = tkinter.Scale(f1, from_ =0, to= 1.00, resolution = 0.01, orient = 'horizontal')
FSat.set(1.0)
FSatB = tkinter.Button(f1, text = "Set saturation", command = lambda: callBack(Action.CHANGE_SAT_FIELD, FSat.get()))


### Vector
VShow = tkinter.Button(f2, text = "Show Vectors",command=lambda: callBack(Action.DRAW_VECS))

VectorColoringDict = {'Black and White': Action.COLOR_MAG_BLACK,
                      'Rainbow': Action.COLOR_MAG_RAINBOW,
                      'Twotone': Action.COLOR_MAG_TWOTONE,
                      'White'  : Action.COLOR_MAG_WHITE
                      }
v_color = ttk.Labelframe(f2, text='Color')
vectClaMinSlider = tkinter.Scale(v_color, from_=0.0, to = 4.99, resolution=0.001, orient='horizontal')
vectClaMinSlider.set(0.0)
vectClaMinButton = tkinter.Button(v_color, text="Set vect clamp min", command=lambda: callBack(Action.SET_VECT_CLAMP_MIN, vectClaMinSlider.get()))
vectClaMaxSlider = tkinter.Scale(v_color, from_=0.02, to = 5.00, resolution=0.001, orient='horizontal')
vectClaMaxSlider.set(1.0)
vectClaMaxButton = tkinter.Button(v_color, text="Set vect clamp max", command=lambda: callBack(Action.SET_VECT_CLAMP_MAX, vectClaMaxSlider.get()))

vector_coloring_dropdown = tkinter.StringVar()
vector_coloring_dropdown.set('White')
vector_coloring_dropdown.trace('w', option_changed_vc)
vecColDropdown = tkinter.OptionMenu(v_color, vector_coloring_dropdown, *VectorColoringDict)
ColorDir = tkinter.Button(v_color, text="Color to Magnitude/Direction/Scalar Field", command=lambda: callBack(Action.COLOR_DIR))
v_color.grid(row=1, columnspan=7, sticky='WE', padx=5, pady=5, ipadx=5, ipady=5)

v_type = ttk.Labelframe(f2, text='Type')
B = tkinter.Button(v_type, text="Change number of vectors (16^2)", command=lambda: callBack(Action.GLYPH_CHANGE_N))
v_typeButton = tkinter.Button(v_type, text="Change vector type", command=lambda: callBack(Action.GLYPH_CHANGE))
v_type.grid(row=1, column = 10, columnspan=7, sticky='WE', padx=5, pady=5, ipadx=5, ipady=5)

VScale = tkinter.Scale(f2, from_=0.01, to=2.00, resolution=0.01, orient='horizontal')
VScale.set(1.00)
VScaleB = tkinter.Button(f2, text="Set Scale", command=lambda: callBack(Action.SET_SCALE_VECTOR, VScale.get()))

VHue = tkinter.Scale(v_color, from_ =0, to= 5, orient = 'horizontal')
VHue.set(0)
VHueB = tkinter.Button(v_color, text = "Set hue", command = lambda: callBack(Action.CHANGE_HUE_VECT, VHue.get()))
streamlinelengthSlider = tkinter.Scale(v_type, from_=1, to=20, resolution=1, orient='horizontal')
streamlinelengthSlider.set(5)
streamlinelengthSliderB = tkinter.Button(v_type,
                                         text="Set Streamline Length",
                                         command=lambda: callBack(Action.SET_STREAMLINE_LENGTH, streamlinelengthSlider.get()))

VDisplaceB = tkinter.Button(f2, text="Random seeding On/Off", command=lambda: callBack(Action.DISPLACE))


VSat = tkinter.Scale(v_color, from_ =0, to= 1.00, resolution = 0.01, orient = 'horizontal')
VSat.set(1.0)
VSatB = tkinter.Button(v_color, text = "Set saturation", command = lambda: callBack(Action.CHANGE_SAT_VECT, VSat.get()))

VdatatypeB = tkinter.Button(f2, text = "Show forcefield (now shown: velocities)", command=lambda: callBack(Action.VELO_TO_FORCE))


VNlevels = tkinter.Scale(v_color, from_=2, to=256, orient='horizontal')
VNlevels.set(50)
VNlevelsB = tkinter.Button(v_color, text="Set Number of Colors", command=lambda: callBack(Action.SET_NLEVELS_VECTOR, VNlevels.get()))
VScale.grid(row=3, column = 10, columnspan=7, sticky='WE', padx=5, pady=5, ipadx=5, ipady=5)
VScaleB.grid(row=4, column = 10, columnspan=7, sticky='WE', padx=5, pady=5, ipadx=5, ipady=5)
VShow.grid(row=0, columnspan=7, sticky='WE', padx=5, pady=5, ipadx=5, ipady=5)
VDisplaceB.grid(row=3, columnspan=7, sticky='WE', padx=5, pady=5, ipadx=5, ipady=5)
VdatatypeB.grid(row = 4,  columnspan=7, sticky='WE', padx=5, pady=5, ipadx=5, ipady=5)
B.pack()
ColorDir.pack()
vecColDropdown.pack()
VNlevels.pack()
VNlevelsB.pack()
v_typeButton.pack()
streamlinelengthSlider.pack()
streamlinelengthSliderB.pack()


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

IScale = tkinter.Scale(f3, from_=0.01, to=2.00, resolution=0.01, orient='horizontal')
IScale.set(1.00)
IScaleB = tkinter.Button(f3, text="Set Scale", command=lambda: callBack(Action.SET_SCALE_ISO, IScale.get()))

INlevels = tkinter.Scale(f3, from_=2, to=256, orient='horizontal')
INlevels.set(50)
INlevelsB = tkinter.Button(f3, text="Set Number of Colors", command=lambda: callBack(Action.SET_ILEVELS_FIELD, INlevels.get()))

IShow = tkinter.Button(f3, text = "Show Isolines",command=lambda: callBack(Action.DRAW_ISO))

IHue = tkinter.Scale(f3, from_ =0, to= 5, orient = 'horizontal')
IHue.set(0)
IHueB = tkinter.Button(f3, text = "Set hue", command = lambda: callBack(Action.CHANGE_HUE_ISO, IHue.get()))

ISat = tkinter.Scale(f3, from_ =0, to= 1.00, resolution = 0.01, orient = 'horizontal')
ISat.set(1.0)
ISatB = tkinter.Button(f3, text = "Set saturation", command = lambda: callBack(Action.CHANGE_SAT_ISO, ISat.get()))


isoClaMinSlider = tkinter.Scale(f3, from_=0.0, to = 4.99, resolution=0.001, orient='horizontal')
isoClaMinSlider.set(0.0)
isoClaMinButton = tkinter.Button(f3, text="Set iso clamp min", command=lambda: callBack(Action.SET_ISO_CLAMP_MIN, isoClaMinSlider.get()))
isoClaMaxSlider = tkinter.Scale(f3, from_=0.02, to = 5.00, resolution=0.001, orient='horizontal')
isoClaMaxSlider.set(1.0)
isoClaMaxButton = tkinter.Button(f3, text="Set iso clamp max", command=lambda: callBack(Action.SET_ISO_CLAMP_MAX, isoClaMaxSlider.get()))


visc_frame = ttk.Labelframe(top, text='Viscosity')
visc_up = tkinter.Button(visc_frame, text="Visc up (*5)", command=lambda: callBack(Action.VISC_UP))
visc_down = tkinter.Button(visc_frame, text="Visc down (/5)", command=lambda: callBack(Action.VISC_DOWN))
visc_up.grid(column=0,row=0, columnspan=1, sticky="nsew")
visc_down.grid(column=1,row=0, columnspan=1, sticky="nsew")

DtSlider = tkinter.Scale(top, from_=0.1, to=1.0, resolution=0.001, orient='horizontal')
DtSlider.set(0.4)
DtButton = tkinter.Button(top, text="Set dt", command=lambda: callBack(Action.SET_DT, DtSlider.get()))
FreezeButton = tkinter.Button(top, text="Freeze", command=lambda: callBack(Action.FREEZE))


# Pack everything
nb.pack()
field_data_dropdown.pack()
field_3ddata_dropdown.pack()
field_c_dropdown.pack()
FScale.pack()
FScaleB.pack()
FXlookS.pack()
FXlookB.pack()
FYlookS.pack()
FYlookB.pack()
DtSlider.pack()
DtButton.pack()
FreezeButton.pack()
visc_frame.pack()

fieldClaMinSlider.pack()
fieldClaMinButton.pack()
fieldClaMaxSlider.pack()
fieldClaMaxButton.pack()
vectClaMinSlider.pack()
vectClaMinButton.pack()
vectClaMaxSlider.pack()
vectClaMaxButton.pack()

isoMinSlider.pack()
isoMinButton.pack()
isoMaxSlider.pack()
isoMaxButton.pack()
isoClaMinSlider.pack()
isoClaMinButton.pack()
isoClaMaxSlider.pack()
isoClaMaxButton.pack()
isoNSlider.pack()
isoNButton.pack()
IScale.pack()
IScaleB.pack()

FHeight.pack()
FHeightB.pack()

FHeightScale.pack()
FHeightScaleB.pack()

FNlevels.pack()
FNlevelsB.pack()

INlevels.pack()
INlevelsB.pack()

FHue.pack()
VHue.pack()
IHue.pack()
FHueB.pack()
VHueB.pack()
IHueB.pack()

FSat.pack()
VSat.pack()
ISat.pack()
FSatB.pack()
VSatB.pack()
ISatB.pack()


FShow.pack()
IShow.pack()

isoColDropdown.pack()

F3d.pack()


### Interactive data value support
while True:
    fieldClaMinSlider.configure(to = fieldClaMaxSlider.get()-0.01)
    fieldClaMaxSlider.configure(from_ = fieldClaMinSlider.get()+0.01)
    vectClaMinSlider.configure(to = vectClaMaxSlider.get()-0.01)
    vectClaMaxSlider.configure(from_ = vectClaMinSlider.get()+0.01)
    isoClaMinSlider.configure(to = isoClaMaxSlider.get()-0.01)
    isoClaMaxSlider.configure(from_ = isoClaMinSlider.get()+0.01)
    if velo:
        VdatatypeB.configure(text = 'Show forcefield (now shown: velocities)')
    else:
        VdatatypeB.configure(text = 'Show velocities (now shown: forcefield)')
    if coldir == 0:
        ColorDir.configure(text = 'Color Magnitude (Direction/Scalar Field)')
    elif coldir == 1:
        ColorDir.configure(text = 'Color Direction (Scalar Field/Magnitude)')
    else:
        ColorDir.configure(text = 'Color Scalar Field (Magnitude/Direction)')
    B.configure(text = 'Change number of vectors (' +str(n_glyphs) +'^2)')
    top.update_idletasks()
    top.update()
