from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QFrame, QCheckBox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
from .patch_generator import PatchGenerator
from shapely.geometry import Polygon
from .patch_model import Patch


class PlotWindow(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        # Inherit attributes from parent
        self.superpositions = self.parent().superpositions
        self.patches = self.parent().patches
        self.centerlines = self.parent().centerlines
        self.patch_params = self.parent().patch_params
        self.raster_mesh = self.parent().raster_mesh
        self.x_mesh, self.y_mesh = self.raster_mesh
        self.raster = self.parent().raster
        self.contours = self.parent().contours

        # Get patch arrays from instance
        self.patches_arrays = [[patch.patch_array for patch in group] for group in self.patches]
        self.plot_index = 0
        self.num_plots = len(self.superpositions)

        # figure
        self.figure, self.ax = plt.subplots()
        self.figure.set_constrained_layout(True)

        self.generate_plot(self.plot_index)
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.update_plot()

    def generate_plot(self, index):
        centerline = self.centerlines[index]
        contour = self.contours[index]
        patches = self.patches_arrays[index]

        superpos = self.superpositions[index]
        self.ax.pcolormesh(self.x_mesh, self.y_mesh, np.where(superpos, self.raster, np.nan), cmap="Reds", shading="auto")

        self.ax.plot(centerline[:, 0], centerline[:, 1], "-", color="blue", label="PCA Centerline", linewidth=1.5)
        self.ax.plot(contour[:, 0], contour[:, 1], "--", color="black", label="Contour")
        for patch in patches:
            self.ax.plot(patch[:, 0], patch[:, 1], "g-", alpha=0.7)
        self.ax.plot([], [], "g-", alpha=0.7, label="Patches")

        self.ax.set_aspect("equal")
        self.ax.set_xlabel("E [km]")
        self.ax.set_ylabel("N [km]")
        self.ax.tick_params(axis="x", labelrotation=90)

    def update_plot(self):
        """Updates the displayed plot"""
        self.ax.clear()
        self.generate_plot(self.plot_index)
        self.ax.legend(fontsize="small")
        self.canvas.draw()


class ControlPanel(QWidget):
    def __init__(self, parent, plot_window):
        super().__init__(parent)
        self.plot_window = plot_window
        self.patch_params = self.parent().patch_params
        self.patch_length, self.patch_width, self.sample_dist = self.patch_params
        self.extraction_state = self.parent().extraction_state
        self.flight_pairs = self.parent().flight_pairs
        self.output_dir = self.parent().output_dir

        self.new_patches_poly = []

        # Useful in the case of single band along centerline
        self.patch_generator = PatchGenerator(self.plot_window.superpositions, self.plot_window.raster_mesh, self.plot_window.raster, self.patch_params)

        self.initUI_panel()

        # single band mode
        self.single_band_mode = False
        self.new_patches_instance = None


    def initUI_panel(self):
        self.setFixedWidth(250)

        # Layout
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # Input fields for parameters
        self.length_label = QLabel("Patch Length:")
        self.length_input = QLineEdit(f"{self.patch_params[0]}")

        self.width_label = QLabel("Patch Width:")
        self.width_input = QLineEdit(f"{self.patch_params[1]}")

        self.distance_label = QLabel("Sample Distance:")
        self.distance_input = QLineEdit(f"{self.patch_params[2]}")

        # Update Button
        self.update_button = QPushButton("Update Fields")
        self.update_button.clicked.connect(self.update_fields)

        # Extraction
        self.extract_button = QPushButton("Extract Patches")
        self.extract_button.clicked.connect(self.proceed_extraction)

        # Navigation buttons
        self.prev_button = QPushButton("<")
        self.prev_button.clicked.connect(self.previous_plot)
        self.next_button = QPushButton(">")
        self.next_button.clicked.connect(self.next_plot)

        # Checkbox
        self.checkBox = QCheckBox("Single band along centerline")
        self.checkBox.stateChanged.connect(self.toggle_band_mode)

        self.flight_label = QLabel(f"Flight pairs: {self.flight_pairs[self.plot_window.plot_index]}")
        layout.addWidget(self.flight_label)

        # Divider Line
        self.layout_dividerLine(layout)

        layout.addWidget(self.length_label)
        layout.addWidget(self.length_input)
        layout.addWidget(self.width_label)
        layout.addWidget(self.width_input)
        layout.addWidget(self.distance_label)
        layout.addWidget(self.distance_input)
        layout.addWidget(self.checkBox)

        self.layout_dividerLine(layout)

        layout.addWidget(self.update_button)
        self.layout_dividerLine(layout)

        # Showing output directory
        self.output_label = QLabel(f"Output folder: {self.output_dir.split('/')[-1]}")
        layout.addWidget(self.output_label)

        layout.addStretch()

        self.layout_dividerLine(layout)

        layout.addWidget(self.extract_button)

        self.layout_dividerLine(layout)

        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        layout.addLayout(nav_layout)

        self.setLayout(layout)

    def layout_dividerLine(self, layout):
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(divider)

    def toggle_band_mode(self, state):
        """Enable/Disable sampling distance and patch length based on checkbox state."""
        if state == 2:  # Checkbox is checked
            self.length_input.setDisabled(True)
            self.distance_input.setDisabled(True)
            self.patch_length = None  # Not used
            self.sample_dist = None  # Not used
            self.single_band_mode = True
        else:
            self.length_input.setDisabled(False)
            self.distance_input.setDisabled(False)
            self.single_band_mode = False
        # update the patch shown on UI
        self.update_all_patches()

    def update_fields(self):
        self.patch_length = int(self.length_input.text())
        self.patch_width = int(self.width_input.text())
        self.sample_dist = int(self.distance_input.text())
        self.patch_params = (self.patch_length, self.patch_width, self.sample_dist)
        print(f"Updated Values - Length: {self.patch_length}, Width: {self.patch_width}, Distance: {self.sample_dist}")

        self.update_all_patches()

    def update_flight_label(self):
        self.flight_label.setText(f"Flight pairs: {self.flight_pairs[self.plot_window.plot_index]}")

    def update_all_patches(self):
        # refaire les patch pour tous toutes les combinaisons
        if not self.single_band_mode:
            new_patch_gen = PatchGenerator(
                superpos_zones=self.plot_window.superpositions,
                raster_mesh=self.plot_window.raster_mesh,
                raster=self.plot_window.raster,
                patch_params=self.patch_params,
            )
            patches_instance = new_patch_gen.patches_list
        else:
            patches_instance = []
            for i in range(self.plot_window.num_plots):
                start_point, max_length = self.patch_generator.compute_max_patch_length(i)
                patch = self.patch_generator.create_single_patch(i, start_point, max_length, self.patch_width)
                patches_instance.append(patch)

        patches_arrays = [[patch.patch_array for patch in group] for group in patches_instance]  # np.arrays patches
        self.plot_window.patches_arrays = patches_arrays
        self.plot_window.update_plot()
        self.new_patches_instance = patches_instance

    def proceed_extraction(self):
        self.update_all_patches()
        self.extraction_state = True  # Update state to break main loop
        self.window().close() 

    def previous_plot(self):
        self.plot_window.plot_index = (self.plot_window.plot_index - 1) % self.plot_window.num_plots
        self.plot_window.update_plot()
        self.update_flight_label()

    def next_plot(self):
        self.plot_window.plot_index = (self.plot_window.plot_index + 1) % self.plot_window.num_plots
        self.plot_window.update_plot()
        self.update_flight_label()


class GUIMainWindow(QMainWindow):
    def __init__(self, superpositions, patches: Patch, centerlines, patch_params, raster_mesh, raster, contours, extraction_state, flight_pairs, output_dir):
        super().__init__()

        self.setWindowTitle("Patch Plotter UI")
        self.setGeometry(100, 100, 900, 500)

        self.superpositions = superpositions
        self.patches = patches
        self.centerlines = centerlines
        self.patch_params = patch_params
        self.raster_mesh = raster_mesh
        self.raster = raster
        self.contours = contours
        self.extraction_state = extraction_state
        self.flight_pairs = flight_pairs
        self.output_dir = output_dir

        self.initUI()
        self.new_patches_poly = self.control_panel.new_patches_poly
        self.new_patches_instance = self.control_panel.new_patches_instance

        
    def initUI(self):

        self.plot_window = PlotWindow(self)
        self.control_panel = ControlPanel(self, self.plot_window)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.control_panel)  
        main_layout.addWidget(self.plot_window)  

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
