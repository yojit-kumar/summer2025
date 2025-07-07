import sys
import os
import tkinter as tk
from tkinter import filedialog, messagebox

class Menu:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Menu")
        self.root.geometry("800x600")

        self.left_section = tk.Frame(self.root, bg="#2a6f5f", width=100, height=200)
        self.left_section.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.right_section = tk.Frame(self.root, bg="#2a6f5f", width=200, height=200)
        self.right_section.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.top_section = tk.Frame(self.root, bg="#2a6f5f", width=200, height=20)
        self.top_section.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.main_frame = tk.Frame(self.root, bg="#2a6f5f")
        self.main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.avatar_label = tk.Label(self.main_frame, text="System", font=("Arial", 10), bg="#2a6f5f")
        self.avatar_label.pack(fill=tk.X, padx=10, pady=10)

        self.system_label = tk.Label(self.main_frame, text="System", font=("Arial", 10), bg="#2a6f5f")
        self.system_label.pack(fill=tk.X, padx=10, pady=10)

        self.options_button = tk.Button(self.main_frame, text="Options", command=self.open_options, font=("Arial", 10), bg="#2a6f5f")
        self.options_button.pack(fill=tk.X, padx=10, pady=10)

        self.quit_button = tk.Button(self.main_frame, text="Quit", command=self.root.destroy, font=("Arial", 10), bg="#2a6f5f")
        self.quit_button.pack(fill=tk.X, padx=10, pady=10)

    def open_options(self):
        options_window = tk.Toplevel(self.root)
        options_window.title("Options")

        # Create options window
        # ...

        options_window.destroy()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    menu = Menu()
    menu.run()
