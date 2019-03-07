import socket
from fluid_actions import Action
import tkinter

top = tkinter.Tk()


def callBack(action):
    clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clientsocket.connect(('localhost', 8089))
    clientsocket.send(action.name.encode('utf-8'))
    print(action.name.encode('utf-8'))




B = tkinter.Button(top, text ="Hello", command = lambda: callBack(Action.GLYPH_CHANGE_N))

S = tkinter.Scale(top, from_=0.00, to=2.00, resolution = 0.01)
S.set(1.00)
Sb = tkinter.Button(top, text = "Set Scale", command = lambda: callBack(Action.FREEZE))


B.pack()
S.pack()
Sb.pack()
top.mainloop()
