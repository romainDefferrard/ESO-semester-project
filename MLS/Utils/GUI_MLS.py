from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import RectangleSelector
from matplotlib.colors import to_rgba
import geopandas as gpd
import csv
import os 


class PlotWindow(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.gdf = self.parent().gdf
        self.intersections = self.parent().intersections
        self.control_panel = None
        self.gdf_filtered = None
        self.current_ids = []
        self.plot_mode = "default"  # "default" or "zoomed"

        # figure
        self.figure, self.ax = plt.subplots()
        self.figure.set_constrained_layout(True)
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.plot_intersections()

    def plot_intersections(self):
        all_ids = set()

        def draw_plot(xmin=None, xmax=None, ymin=None, ymax=None):
            self.ax.clear()
            all_ids.clear()
            ids_pairs = []

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
                num_ids = len(ids)
                base_cmap = plt.colormaps["tab20"]
                colors = [base_cmap(j / num_ids) for j in range(num_ids)]
                ids_pairs.append((row["id_1"], row["id_2"]))

                for idx, seg_id in enumerate(ids):
                    if seg_id in all_ids:
                        continue
                    all_ids.add(seg_id)
                    seg = overlapping_segs[overlapping_segs["id"] == seg_id]
                    base_color = colors[idx]
                    fill_color = to_rgba(base_color, alpha=0.3)
                    edge_color = to_rgba(base_color, alpha=0.9)

                    seg["buffer"].plot(ax=self.ax, color=fill_color, edgecolor=edge_color, linewidth=1)
                    seg["geometry"].plot(ax=self.ax, color=edge_color, linewidth=1.2)

            if not intersections_filtered.empty:
                intersections_filtered["overlap_geom"].plot(ax=self.ax, color="none", edgecolor="red", linewidth=2)

            self.ax.set_title("All Intersections and Segments")
            self.ax.set_aspect("equal")
            # ax.legend(handles=legend_handles, loc="upper right", fontsize="small")
            self.canvas.draw()
            self.current_ids = ids_pairs

            if len(self.current_ids) > 20:
                return  # Too many segments to display
            if self.control_panel:
                self.control_panel.displayed_ids.setText(
                    "\n".join([f"{a}, {b}" for a, b in self.current_ids]) if self.plot_mode == "zoomed" else "All Segments"
                )
                self.control_panel.update_intersection_count(len(intersections_filtered))

        draw_plot()

        toggle_selector = RectangleSelector(
            self.ax,
            None,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )

        def onselect(eclick, erelease):
            nonlocal toggle_selector

            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            xmin, xmax = sorted([x1, x2])
            ymin, ymax = sorted([y1, y2])
            if abs(xmax - xmin) > 1e-3 and abs(ymax - ymin) > 1e-3:
                # Destroy and reinstantiate selector to remove the rectangle
                toggle_selector.disconnect_events()
                toggle_selector.set_visible(False)
                toggle_selector = RectangleSelector(self.ax, onselect, useblit=True, button=[1], minspanx=5, minspany=5, spancoords="pixels", interactive=True)
                draw_plot(xmin, xmax, ymin, ymax)

        toggle_selector.set_active(True)
        toggle_selector.onselect = onselect

        def on_click(event):
            if event.inaxes == self.ax:
                x, y = event.xdata, event.ydata
                point = gpd.GeoSeries([gpd.points_from_xy([x], [y])[0]], crs=self.intersections.crs)
                matches = self.intersections[self.intersections.intersects(point[0])]
                if not matches.empty:
                    for _, row in matches.iterrows():
                        if self.control_panel:
                            self.control_panel.clicked_id_label.setText(f"<span style='color: red;'>Clicked ID: {row['id_1']}, {row['id_2']}</span>")

        self.canvas.mpl_connect("button_press_event", on_click)


class ControlPanel(QWidget):
    def __init__(self, parent, plot_window):
        super().__init__(parent)
        self.plot_window = plot_window
        self.intersections = self.parent().intersections
        self.output_path = self.parent().output_path

        self.initUI_panel()

    def initUI_panel(self):
        self.setFixedWidth(250)
        layout = QVBoxLayout()
        layout.setSpacing(10)
        self.setLayout(layout)

        layout.addWidget(QLabel("Intersection Viewer"))
        self.layout_dividerLine(layout)

        self.intersections_count_label = QLabel(f"Total Intersections: {len(self.intersections)}")
        layout.addWidget(self.intersections_count_label)

        self.ids_label = QLabel("Displayed IDs")
        layout.addWidget(self.ids_label)

        self.displayed_ids = QLabel(" ")
        layout.addWidget(self.displayed_ids)

        layout.addStretch()

        self.clicked_id_label = QLabel("Clicked Intersection Info")
        layout.addWidget(self.clicked_id_label)

        self.extract_button = QPushButton("Extract Segments")
        self.extract_button.clicked.connect(self.extract_segments)
        layout.addWidget(self.extract_button)

        self.layout_dividerLine(layout)

        self.reset_button = QPushButton("Reset Plot")
        self.reset_button.clicked.connect(self.plot_window.plot_intersections)
        layout.addWidget(self.reset_button)

    def update_intersection_count(self, count):
        self.intersections_count_label.setText(f"Intersections displayed: {count}")

    def layout_dividerLine(self, layout):
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(divider)

    def extract_segments(self):

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

        self.extraction_state = True  # Update state to break main loop
        self.window().close()


class GUI_MLS(QMainWindow):
    def __init__(self, gdf, intersections, output):
        super().__init__()

        self.setWindowTitle("MLS UI")
        self.setGeometry(100, 100, 900, 500)

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
