import socket
from fluid_actions import Action
import tkinter
import numpy as np
top = tkinter.Tk()


def callBack(action, value = None):
    clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clientsocket.connect(('localhost', 8089))
    message = action.name
    if value is not None:
        message += ':' + str(value)
    clientsocket.send(message.encode('utf-8'))
    print(action.name.encode('utf-8'))




B = tkinter.Button(top, text ="Hello", command = lambda: callBack(Action.GLYPH_CHANGE_N))
ColorDir = tkinter.Button(top, text ="Color Dir", command = lambda: callBack(Action.COLOR_DIR))
VecColList = {'vector coloring bw' : Action.COLOR_MAG_BLACK, 'vector coloring rainbow' : Action.COLOR_MAG_RAINBOW, 'vector coloring twotone' : Action.COLOR_MAG_TWOTONE}
vc = tkinter.StringVar()
vc.set('vector coloring bw')
VecCol = tkinter.OptionMenu(top, vc ,*VecColList)

vc.trace('w', callBack(VecColList[vc.get()]) )

S = tkinter.Scale(top, from_=0.00, to=2.00, resolution = 0.01)
S.set(1.00)
Sb = tkinter.Button(top, text = "Set Scale", command = lambda: callBack(Action.SET_SCALE, S.get()))
DtSlider = tkinter.Scale(top, from_ = 0.1, to = 1.0, resolution =  0.001)
DtSlider.set(0.4)
DtButton = tkinter.Button(top, text = "Set dt", command = lambda: callBack(Action.SET_DT, DtSlider.get()))




B.pack()
ColorDir.pack()
VecCol.pack()
S.pack()
Sb.pack()
DtSlider.pack()
DtButton.pack()
top.mainloop()
