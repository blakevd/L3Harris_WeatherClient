import tkinter as tk
from tkinter import ttk
from tkinter import Frame
from controller import select
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import us
import tkinter
import tkintermapview

class WeatherGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Weather Prediction GUI")
        root.configure(bg="lightgray")

        # Get the screen width and height
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # Calculate the dimensions for 1/4 of the screen
        window_width = screen_width // 2
        window_height = screen_height // 2

        # Set the window size and position
        self.root.geometry(f"{window_width}x{window_height}+{screen_width//4}+{screen_height//4}")

        frame = tk.Frame(root)
        frame.pack(fill=tk.BOTH, expand=True)

        self.listbox = tk.Listbox(frame, bg="lightgray", fg="black", font=("Helvetica", 12),
                                  highlightbackground="lightgray", highlightthickness=1, width=30, activestyle="none")
        for state in us.states.STATES:
            self.listbox.insert(tk.END, " > " + state.name)
        self.listbox.pack(side=tk.LEFT, fill=tk.Y, expand=False)
        self.listbox.bind("<<ListboxSelect>>", self.update_stations)

        self.map_widget = tkintermapview.TkinterMapView(frame, width=800, height=600, corner_radius=0)
        self.map_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.map_widget.set_position(48.860381, 2.338594)  # Paris, France
        self.map_widget.set_zoom(15)
        
        marker_1 = self.map_widget.set_marker(48.860381, 2.338594, text="Brandenburger Tor")


    def update_stations(self, event):
        selected_index = self.listbox.curselection()

        # Check if there is a selected index
        if selected_index:
            # Get the selected state directly from the listbox
            selected_state = self.listbox.get(selected_index)

            if selected_state.startswith(" > "):
                selected_state = selected_state[3:]
                # Assuming self.stations is a list of stations for the selected state
                self.stations = get_stations_for_state(selected_state)

                scrollbar_position = self.listbox.yview()[0]
                # Clear the current state listbox
                self.listbox.delete(0, tk.END)

                # Re-populate the state listbox with the states and append stations under the selected state
                for state in us.states.STATES:
                    state_name = state.name
                    self.listbox.insert(tk.END, " > " + state_name)
                    if state_name == selected_state:
                        for station in self.stations:
                            # Insert stations with a special tag "station"
                            self.listbox.insert(tk.END, "    - " + station)
                self.listbox.yview_moveto(scrollbar_position)
            elif selected_state.startswith("    - "):
                selected_state = selected_state[6:]





    def update_listbox(self):

        # Get the selected station ID
        selected_station = self.station_var.get()

        # Run the select query and get the results
        results = run_select_query('station_name', selected_station)
        
        # Extract temperature and last update time for plotting
        temps = []
        last_update_times = []

        # Loop through each result and extract temperature
        count = 0
        for result in results:
            temps.append(float(result.temp_f))
            last_update_times.append(count)
            count = count + 1


        # Plotting using Matplotlib
        fig, ax = plt.subplots()
        ax.plot(last_update_times, temps, marker='o')
        ax.set_title(f'Temperature Trends for {selected_station}')
        ax.set_xlabel('Last Update Time')
        ax.set_ylabel('Temperature (°F)')

    # Embed the Matplotlib plot in the Tkinter window using Canvas
        if hasattr(self, 'canvas_frame'):
            self.canvas_frame.destroy()

        self.canvas_frame = tk.Frame(self.plot_frame)
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Example function to get stations based on the selected state
def get_stations_for_state(state):
    weather_data_array = run_select_query('state', state)
    # Extract 'station name' from each WeatherData instance
    stations = list(set(weather_data.station_name for weather_data in weather_data_array))
    return stations

def run_select_query(column, constraint):
    return select(table_col=column, col_constraint=constraint)



# Main code to run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = WeatherGUI(root)
    root.mainloop()
