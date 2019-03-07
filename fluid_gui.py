import socket
from fluid_actions import Action
import tkinter

top = tkinter.Tk()


def helloCallBack():
    clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clientsocket.connect(('localhost', 8089))
    action = Action.GLYPH_CHANGE_N
    clientsocket.send(action.name.encode('utf-8'))
    print(action.name.encode('utf-8'))


B = tkinter.Button(top, text ="Hello", command = helloCallBack)

B.pack()
top.mainloop()