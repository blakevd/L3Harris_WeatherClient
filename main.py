# main.py
import tkinter as tk
from view import WeatherGUI

if __name__ == '__main__':
    root = tk.Tk()
    gui = WeatherGUI(root)
    gui.root.mainloop()