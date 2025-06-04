"""
Filename: GUI_MLS.py
Author: Romain Defferrard
Date: 04-06-2025

Description:
    PyQt6 interface for Mobile Laser Scanning (MLS) segment intersection analysis.
    Includes:
        - PlotWindow: interactive matplotlib display of buffer geometries and intersections.
        - ControlPanel: extraction, reset, and segment information display panel.
        - GUI_MLS: main window class to integrate plotting and control components.
    This pipeline uses a TimerLogger utility to benchmark steps in the pipeline.
"""


from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import RectangleSelector
from matplotlib.colors import to_rgba
import geopandas as gpd
import csv
import os
import logging
from .timer_logger import TimerLogger



class PlotWindow(QWidget):
    def __init__(self, parent):
        """
        Widget for plotting intersections and buffers over segments using matplotlib.
        Allows rectangular selection for zooming and filtering.

        Input:
            parent (QWidget): Parent main window (GUI_MLS)

        Output:
            None
        """
        super().__init__(parent)
        self.gdf = self.parent().gdf
        self.intersections = self.parent().intersections
        self.control_panel = None
        self.gdf_filtered = None
        self.current_ids = []
        self.plot_mode = "default"  # "default" or "zoomed"

        self.figure, self.ax = plt.subplots()
        self.figure.set_constrained_layout(True)
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.toggle_selector = None
        self.plot_intersections()
        self.setup_selector()

    def setup_selector(self):
        """
        Initialize interactive rectangular selector for zooming.

        Output:
            None
        """
        self.toggle_selector = RectangleSelector(
            self.ax,
            self.on_select,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )
        self.toggle_selector.set_active(True)
        self.canvas.mpl_connect("button_press_event", self.on_click)

    def plot_intersections(self, xmin=None, xmax=None, ymin=None, ymax=None):
        """
        Plot buffer and geometry data for overlapping segment zones. 
        
        Inputs:
            xmin, xmax, ymin, ymax (float): Optional zoom coordinates

        Output:
            None
        """
        all_ids = set()
        ids_pairs = []

        self.ax.clear()
        if self.control_panel:
            self.control_panel.displayed_ids.setText("")
            self.control_panel.clicked_id_label.setText("Clicked Intersection Info")

        if None not in (xmin, xmax, ymin, ymax):
            self.gdf_filtered = self.gdf.cx[xmin:xmax, ymin:ymax]
            intersections_filtered = self.intersections.cx[xmin:xmax, ymin:ymax]
            self.plot_mode = "zoomed"
        else:
            self.gdf_filtered = self.gdf
            intersections_filtered = self.intersections
            self.plot_mode = "default"

        for i, row in intersections_filtered.iterrows():
            zone_geom = row["overlap_geom"]
            overlapping_segs = self.gdf_filtered[self.gdf_filtered["buffer"].intersects(zone_geom)]
            ids = overlapping_segs["id"].unique()
            #num_ids = len(ids)
            base_cmap = plt.colormaps["tab20"]
            colors = {seg_id: base_cmap((hash(seg_id) % 20) / 20) for seg_id in ids}
            ids_pairs.append((row["id_1"], row["id_2"]))

            for idx, seg_id in enumerate(ids):
                if seg_id in all_ids:
                    continue
                all_ids.add(seg_id)
                seg = overlapping_segs[overlapping_segs["id"] == seg_id]
                base_color = colors[seg_id]

                fill_color = to_rgba(base_color, alpha=0.3)
                edge_color = to_rgba(base_color, alpha=0.9)

                seg["buffer"].plot(ax=self.ax, color=fill_color, edgecolor=edge_color, linewidth=1)
                seg["geometry"].plot(ax=self.ax, color=edge_color, linewidth=1.2)

        if not intersections_filtered.empty:
            intersections_filtered["overlap_geom"].plot(ax=self.ax, color="none", edgecolor="red", linewidth=2)

        self.ax.set_title("Intersection from shp segments")
        self.ax.set_aspect("equal")
        self.canvas.draw()

        self.current_ids = ids_pairs
         
        
        if self.control_panel:
            self.control_panel.update_intersection_count(len(intersections_filtered))
            if len(self.current_ids) > 30:
                return
            self.control_panel.displayed_ids.setText(
                "\n".join([f"{a}, {b}" for a, b in self.current_ids])
                if self.plot_mode == "zoomed" else "All Segments"
            )
            self.control_panel.update_intersection_count(len(intersections_filtered))

    def on_select(self, eclick, erelease):
        """
        Callback for rectangular zoom selector.

        Output:
            None
        """
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        if abs(xmax - xmin) > 1e-3 and abs(ymax - ymin) > 1e-3:
            self.toggle_selector.set_visible(False)
            self.toggle_selector.disconnect_events()
            self.plot_intersections(xmin, xmax, ymin, ymax)
            self.setup_selector()

    def on_click(self, event):
        """
        Displays clicked intersection segment ID in the control panel.

        Output:
            None
        """
        if event.inaxes == self.ax:
            x, y = event.xdata, event.ydata
            point = gpd.GeoSeries([gpd.points_from_xy([x], [y])[0]], crs=self.intersections.crs)
            matches = self.intersections[self.intersections.intersects(point[0])]
            if not matches.empty and self.control_panel:
                for _, row in matches.iterrows():
                    self.control_panel.clicked_id_label.setText(
                        f"<span style='color: red;'>Clicked ID: {row['id_1']}, {row['id_2']}</span>"
                    )
            


class ControlPanel(QWidget):
    def __init__(self, parent, plot_window):
        """
        Control panel widget with buttons and labels for interaction and export.

        Input:
            parent (QWidget): Parent window
            plot_window (PlotWindow): Associated plot widget

        Output:
            None
        """
        super().__init__(parent)
        self.plot_window = plot_window
        self.intersections = self.parent().intersections
        self.output_path = self.parent().output_path
        self.initUI_panel()
        self.timer = TimerLogger()
        logging.basicConfig(level=logging.INFO, format='%(message)s')


    def initUI_panel(self):
        """
        Initialize layout and interactive controls.

        Output:
            None
        """
        self.setFixedWidth(250)
        layout = QVBoxLayout()
        layout.setSpacing(10)
        self.setLayout(layout)

        self.layout_dividerLine(layout)

        self.intersections_count_label = QLabel(f"Total Intersections: {len(self.intersections)}")
        layout.addWidget(self.intersections_count_label)

        self.ids_label = QLabel("Displayed IDs (max 30)")
        layout.addWidget(self.ids_label)
        
        self.layout_dividerLine(layout)


        self.displayed_ids = QLabel(" ")
        layout.addWidget(self.displayed_ids)

        layout.addStretch()

        self.clicked_id_label = QLabel("No intersection selected")
        layout.addWidget(self.clicked_id_label)

        self.extract_button = QPushButton("Extract Segments")
        self.extract_button.clicked.connect(self.extract_segments)
        layout.addWidget(self.extract_button)

        self.layout_dividerLine(layout)

        self.reset_button = QPushButton("Reset Plot")
        self.reset_button.clicked.connect(self.reset_plot)
        layout.addWidget(self.reset_button)

    def reset_plot(self):
        """
        Resets the plot to full extent.

        Output:
            None
        """
        self.plot_window.plot_intersections()
        self.plot_window.setup_selector()

    def update_intersection_count(self, count):
        """
        Update the displayed intersection count label.

        Input:
            count (int): Number of intersections

        Output:
            None
        """
        self.intersections_count_label.setText(f"Intersections displayed: {count}")

    def layout_dividerLine(self, layout):
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(divider)

    def extract_segments(self):
        """
        Export the currently displayed segment pairs to a CSV file.
        """
        self.timer.start("Extraction")
        intersection_ids = self.plot_window.current_ids
        gdf = self.plot_window.gdf_filtered
        rows = []
        for id1, id2 in intersection_ids:
            name1 = gdf[gdf["id"] == id1]["name"].values[0]
            name2 = gdf[gdf["id"] == id2]["name"].values[0]
            rows.append((id1, id2, name1, name2))

        filename = self.output_path
        base, ext = os.path.splitext(self.output_path)
        filename = f"{base}_{len(rows)}{ext}"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["id_1", "id_2", "name_1", "name_2"])
            writer.writerows(rows)

        print(f"Extracted {len(rows)} segment pairs to {filename}")
        self.extraction_state = True
        self.window().close()
        self.timer.stop("Extraction")
        self.timer.summary()


class GUI_MLS(QMainWindow):
    def __init__(self, gdf, intersections, output):
        """
        Main GUI window for displaying MLS intersections.

        Inputs:
            gdf (GeoDataFrame): Segment geometries with buffer
            intersections (GeoDataFrame): Precomputed intersection zones
            output (str): Output path for CSV export

        Output:
            None
        """
        super().__init__()
        self.setWindowTitle("MLS UI")
        self.setGeometry(100, 100, 1100, 700)

        self.gdf = gdf
        self.intersections = intersections
        self.output_path = output
        self.initUI()

    def initUI(self):
        self.plot_window = PlotWindow(self)
        self.control_panel = ControlPanel(self, self.plot_window)
        self.plot_window.control_panel = self.control_panel

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.control_panel)
        main_layout.addWidget(self.plot_window)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)