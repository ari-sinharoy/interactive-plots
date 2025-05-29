# -*- coding: utf-8 -*-
"""
Created on Thu May 29 12:37:52 2025

@author: as836
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LinearRegression

class InteractivePlotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive Data Plotter")

        self.load_button = tk.Button(root, text="Load Data (.csv/.txt)", command=self.load_data)
        self.load_button.pack(pady=5)

        self.subtract_button = tk.Button(root, text="Subtract Fit Line", command=self.subtract_line, state=tk.DISABLED)
        self.subtract_button.pack(pady=5)

        self.save_button = tk.Button(root, text="Save Subtracted Data", command=self.save_plot, state=tk.DISABLED)
        self.save_button.pack(pady=5)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_click)

        self.data = None
        self.selected_points = []
        self.fitted_model = None
        self.subtracted_plot_ready = False

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV or TXT", "*.csv *.txt")])
        if not file_path:
            return
        try:
            if file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_csv(file_path, delim_whitespace=True, header=None)
            df.columns = ['x', 'y']
            self.data = df
            self.selected_points = []
            self.fitted_model = None
            self.subtracted_plot_ready = False
            self.plot_data()
            self.subtract_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.DISABLED)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data:\n{e}")

    def plot_data(self):
        self.ax.clear()
        self.ax.plot(self.data['x'], self.data['y'], label='Original Data')
        self.ax.set_title("Click to select, double-click to unselect")
        self.ax.legend()
        self.canvas.draw()

    def on_click(self, event):
        if self.data is None or event.inaxes != self.ax:
            return

        x_click, y_click = event.xdata, event.ydata
        if x_click is None or y_click is None:
            return

        y_min, y_max = self.data['y'].min(), self.data['y'].max()

        #if event.dblclick:
        #    self.remove_nearest_point(x_click, y_click)
        #    self.redraw_plot()
        if event.button == 3:  # Right-click to remove
            self.remove_nearest_point(x_click, y_click)
            self.redraw_plot()
        else:
            # Only allow selection if y is within valid range
            if y_min <= y_click <= y_max:
                self.selected_points.append((x_click, y_click))
                self.redraw_plot()

    def remove_nearest_point(self, x_click, y_click):
        if not self.selected_points:
            return

        dist_thresh = 0.05 * (self.ax.get_xlim()[1] - self.ax.get_xlim()[0])
        closest_idx = None
        min_dist = float('inf')
        for i, (x, y) in enumerate(self.selected_points):
            dist = np.hypot(x - x_click, y - y_click)
            if dist < min_dist and dist < dist_thresh:
                closest_idx = i
                min_dist = dist
        if closest_idx is not None:
            del self.selected_points[closest_idx]

    def redraw_plot(self):
        self.ax.clear()
        self.ax.plot(self.data['x'], self.data['y'], label='Original Data')

        if self.selected_points:
            for x, y in self.selected_points:
                self.ax.plot(x, y, 'ro')
                self.ax.annotate(f'({x:.2f}, {y:.2f})', (x, y), textcoords="offset points",
                                 xytext=(5, 5), ha='left', fontsize=8)

            if len(self.selected_points) >= 2:
                points = np.array(self.selected_points)
                x_vals = points[:, 0].reshape(-1, 1)
                y_vals = points[:, 1]
                model = LinearRegression()
                model.fit(x_vals, y_vals)
                self.fitted_model = model
                x_line = np.linspace(self.data['x'].min(), self.data['x'].max(), 500)
                y_line = model.predict(x_line.reshape(-1, 1))
                self.ax.plot(x_line, y_line, 'g--', label='Fit Line')
            else:
                self.fitted_model = None

        self.ax.set_title("Click to select, double-click to unselect")
        self.ax.legend()
        self.canvas.draw()

    def subtract_line(self):
        if not self.fitted_model or self.data is None:
            messagebox.showwarning("Warning", "Select at least two points to define a line first.")
            return

        x_vals = self.data['x'].values.reshape(-1, 1)
        y_fit = self.fitted_model.predict(x_vals)
        self.data['y_subtracted'] = self.data['y'] - y_fit

        self.ax.clear()
        self.ax.plot(self.data['x'], self.data['y_subtracted'], label="Subtracted Data", color='orange')
        self.ax.set_title("Data after Subtracting Fit Line")
        self.ax.legend()
        self.canvas.draw()

        self.subtracted_plot_ready = True
        self.save_button.config(state=tk.NORMAL)

    def save_plot(self):
        if not self.subtracted_plot_ready or 'y_subtracted' not in self.data.columns:
            messagebox.showinfo("Info", "No subtracted data to save yet.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                 filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if file_path:
            try:
                self.data[['x', 'y_subtracted']].to_csv(file_path, sep='\t', index=False, header=False, float_format='%.8f')
                messagebox.showinfo("Saved", f"Subtracted data saved to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save data:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = InteractivePlotApp(root)
    root.mainloop()