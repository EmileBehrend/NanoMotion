import argparse
import hashlib
import importlib
import json
import os
import platform
import sys

import PyQt5
import h5py
import hdf5plugin
import numpy as np
import pims
import pyqtgraph
import pyqtgraph as pg
import pyqtgraph.exporters
import skimage.color
import skimage.exposure
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.uic import loadUiType

import utils
from video_backend import VideoSequence, H5Sequence, PimsSequence

dirname = os.path.dirname(__file__)

Ui_MainWindow, QMainWindow = loadUiType(os.path.join(dirname, "main_menu.ui"))


class Main(QMainWindow, Ui_MainWindow):
    def __init__(self, ):
        super(Main, self).__init__()
        self.setupUi(self)

        self.pg_widget = None
        self.pg_view_box = None
        self.pg_image_item = None

        self.saved_rectangles = {}
        self.rectangles = []  # list of ROIs to analyze
        self.selected = None  # selected rectangle

        self.plots_dict = {}  # list of plots to plot
        self.opened_plots = []

        self.output_basepath = None
        self.basename = None

        self.original_video_length = 0
        self.solver = None

        if platform.system() == "Darwin":
            self.menubar.setNativeMenuBar(False)

        self.actionOpen.triggered.connect(self.browse_files)
        self.actionExport_results.triggered.connect(self.export_results)
        self.actionAdd_box.triggered.connect(self.add_draggable_rectangle)

        self.actionAdd_box = QtWidgets.QAction()
        self.actionAdd_box.setObjectName("actionAdd_box")
        self.menubar.addAction(self.actionAdd_box)
        self.actionAdd_box.setText("Add analysis box")
        self.actionAdd_box.triggered.connect(self.add_draggable_rectangle)
        self.actionAdd_box.setShortcut("A")

        self.actionRemove_box = QtWidgets.QAction()
        self.actionRemove_box.setObjectName("actionRemove_box")
        self.menubar.addAction(self.actionRemove_box)
        self.actionRemove_box.setText("Remove analysis box")
        self.actionRemove_box.triggered.connect(self.remove_rectangle)
        self.actionRemove_box.setShortcut("R")

        self.actionStart_solver = QtWidgets.QAction()
        self.actionStart_solver.setObjectName("actionStart_solver")
        self.menubar.addAction(self.actionStart_solver)
        self.actionStart_solver.setText("Start analysis")
        self.actionStart_solver.triggered.connect(self.start_analysis)
        self.actionStart_solver.setShortcut("S")

        self.actionShow_results = QtWidgets.QAction()
        self.actionShow_results.setObjectName("actionShow_results")
        self.menubar.addAction(self.actionShow_results)
        self.actionShow_results.setText("Show plots")
        self.actionShow_results.triggered.connect(self.show_results)
        self.actionShow_results.setShortcut("V")

        self.actionStop_solver = QtWidgets.QAction()
        self.actionStop_solver.setObjectName("actionStop_solver")
        self.menubar.addAction(self.actionStop_solver)
        self.actionStop_solver.setText("Stop analysis")
        self.actionStop_solver.triggered.connect(self.stop_analysis)

        self.actionReset_boxes = QtWidgets.QAction()
        self.actionReset_boxes.setObjectName("actionReset_boxes")
        self.menubar.addAction(self.actionReset_boxes)
        self.actionReset_boxes.setText("Reset boxes")
        self.actionReset_boxes.triggered.connect(self.reset_boxes)

        self.json_data = {}

        self.file_name = None
        self.id = None
        self.video_data: VideoSequence = None

        self.load_parameters()

        if args.open is not None:
            self.file_name = args.open

        self.cursor = None

        self.load_and_show_file()

        if args.autostart:
            self.start_analysis()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        print("File dropped in the window.")

        if e.mimeData().hasUrls:
            for url in e.mimeData().urls():
                self.file_name = str(url.toLocalFile())

            self.load_and_show_file()

    def set_plot_options(self):
        for action in self.menuView_plot.actions():
            self.plots_dict[action.objectName()] = action.isChecked()
            print(f"Menu option '{action.objectName()}' ('{action.text()}') is set to {action.isChecked()}.")

        self.line_chop_sec.setEnabled(self.view_violin_chop.isChecked())
        self.label_chop_sec.setEnabled(self.view_violin_chop.isChecked())

    def browse_files(self):
        self.file_name, _ = QFileDialog.getOpenFileName(self, "File to analyze", "", "All files (*);;Video files (.mp4 *.avi);;Stacks (*.h5 *.h5s)")
        self.load_and_show_file()

    def unload_file(self):
        self.views.clear()  # clear the image name in the list
        self.rectangles_list.clear()  # clear the list of boxes on the right side

        for rectangle in self.rectangles:
            self.pg_view_box.removeItem(rectangle)

        self.rectangles.clear()

        if self.solver is not None:
            self.solver.clear_annotations()

        self.basename = None

        self.mplvl.removeWidget(self.pg_widget)

    def pg_on_move(self, pos):
        mapped = self.pg_view_box.mapSceneToView(pos)
        self.cursor = (mapped.x(), mapped.y())

    def load_and_show_file(self, load_file=True):
        if self.file_name is None:
            return

        if load_file:
            try:
                if os.path.isfile(self.file_name):
                    _, extension = os.path.splitext(self.file_name)

                    if extension in [".h5", ".h5s"]:
                        self.video_data = H5Sequence(h5py.File(self.file_name))
                    else:
                        self.video_data = PimsSequence(pims.Video(self.file_name))

                    with open(self.file_name, "rb") as stream:
                        self.id = hashlib.blake2b(stream.read()).hexdigest()

                        # TODO: load Fletcher-32 checksums for HDF5 files
                        # https://stackoverflow.com/questions/62946682/accessing-fletcher-32-checksum-in-hdf5-file
                else:
                    self.video_data = PimsSequence(pims.ImageSequence(self.file_name))

                    self.id = self.file_name
            except Exception as e:
                print(f"Failed to load file/folder '{self.file_name}'.")
                print(e)

                if args.autostart and args.quit:  # silent quit if no file was found and automation is running
                    app.quit()

                return

        print(f"Loaded file '{self.file_name}'")
        if self.id in self.json_data["boxes"]:
            self.saved_rectangles = self.json_data["boxes"][self.id]

            print("Loaded previously saved boxes (with blake2b hash).")
        elif self.file_name in self.json_data["boxes"]:  # fallback to filename before giving up
            self.saved_rectangles = self.json_data["boxes"].pop(self.file_name)  # remove previous id

            self.json_data["boxes"][self.id] = self.saved_rectangles  # set new id

            print("Loaded previously saved boxes (with filename).")
        else:
            self.saved_rectangles = {}

        try:
            self.unload_file()
        except AttributeError:
            print("Nothing to clear.")

        shape = np.shape(self.video_data.get_frame(0))
        try:
            # Try to get the video length (can vary depending on the Python environment)
            self.original_video_length = self.video_data.length
        except Exception:
            print("Can't get video length.")

        print(f"Shape of `video_data[0]` object: {shape} * {self.original_video_length} frames, type: {type(self.video_data)}")

        self.views.addItem(self.file_name)

        self.pg_view_box = pg.ViewBox()
        self.pg_view_box.setAspectLocked()
        self.pg_view_box.invertY()

        self.pg_widget = pg.GraphicsView()
        self.pg_widget.setCentralWidget(self.pg_view_box)

        self.pg_image_item = pg.ImageItem(axisOrder="row-major")
        self.pg_view_box.addItem(self.pg_image_item)
        try:
            display = self.video_data.get_frame(int(self.line_start_frame.text()))

            display = skimage.color.rgb2gray(display)

            import filter
            importlib.reload(filter)

            display = filter.filter_image(display, contrast=self.checkbox_contrast.isChecked())

            self.pg_image_item.setImage(display)
            # self.pg_image_item.setImage(display)

            print(f"Image: {display.dtype}")
        except Exception as e:
            print(e)
            self.pg_image_item.setImage(self.video_data.get_frame(0))
            # self.pg_image_item.setImage(self.video_data.get_frame(0))

        self.mplvl.addWidget(self.pg_widget)

        self.pg_image_item.scene().sigMouseMoved.connect(self.pg_on_move)

        for box in self.saved_rectangles.values():
            self.add_rectangle(box["number"], box["x0"], box["y0"], box["width"], box["height"])

    def add_draggable_rectangle(self):
        if self.cursor is None:  # no file opened, return gracefully
            return

        width = int(self.line_w.text())
        height = int(self.line_h.text())

        if self.cursor[0] is not None and self.cursor[1] is not None:
            x0 = self.cursor[0] - width / 2
            y0 = self.cursor[1] - height / 2
        else:
            x0 = width / 2 + 15
            y0 = height / 2 + 15

        number = len(self.rectangles)

        self.add_rectangle(number, x0, y0, width, height)

    def pg_rectangle_selected(self, event):
        # print(f"Selected: {event}")
        if self.selected is not None:
            self.selected.setPen(pg.mkPen())  # set old selected to normal color

        self.selected = event
        self.selected.setPen(pg.mkPen(color="r"))  # highlight new selected rectangle

    def pg_rectangle_remove(self, event):
        self.remove_rectangle(rectangle=event)

    def add_rectangle(self, number, x0, y0, width, height):
        print(f"Adding box {number} to figure.")

        rectangle = pyqtgraph.RectROI((x0, y0), (width, height), scaleSnap=True, translateSnap=True, rotatable=False, removable=True)
        rectangle.setAcceptedMouseButtons(PyQt5.QtCore.Qt.MouseButton.LeftButton)
        # rectangle.sigRegionChangeFinished.connect(self.pg_rectangle_selected)
        rectangle.sigClicked.connect(self.pg_rectangle_selected)
        rectangle.sigRemoveRequested.connect(self.pg_rectangle_remove)
        self.pg_view_box.addItem(rectangle)

        text = pg.TextItem(str(number), color=(255, 0, 0), anchor=(0.25, 0.85))
        text.setParentItem(rectangle)

        font = pg.Qt.QtGui.QFont()
        font.setPixelSize(20)

        text.setFont(font)

        self.rectangles.append(rectangle)
        self.rectangles_list.addItem(str(number))

    def remove_rectangle(self, rectangle=None):
        if len(self.rectangles) <= 0:  # no box present, return gracefully
            return

        if rectangle is None or not rectangle:
            if self.selected is None:
                return
            else:
                rectangle = self.selected

        # print(f"Rectangle: {rectangle}, selected: {self.selected}")

        number = self.rectangles.index(rectangle)
        print(f"Removing box {number} from figure.")

        self.rectangles_list.takeItem(len(self.rectangles) - 1)
        self.rectangles.remove(rectangle)

        self.pg_view_box.removeItem(rectangle)

        self.selected = None

        i = 0
        for r in self.rectangles:
            for child in r.allChildItems():
                if isinstance(child, pg.TextItem):
                    print(f"Type: {type(child)}")
                    child.setText(str(i))

            i += 1

    def get_parameter(self, name: str, default: object) -> object:
        parts = name.split(".")

        data = self.json_data
        while len(parts) > 1:
            key = parts.pop(0)

            data = data[key]

        return data.get(parts[0], default)

    def load_parameters(self):
        with open(os.path.join(dirname, "settings.json"), "r") as json_file:
            self.json_data = json.load(json_file)

            if "last_file" in self.json_data:
                self.file_name = self.json_data["last_file"]

            self.line_pixel_size.setText(str(self.get_parameter("parameters.pixel_size", 1.12)))
            self.line_magn.setText(str(self.get_parameter("parameters.magnification", 0)))
            self.line_sub_pixel.setText(str(self.get_parameter("parameters.sub_pixel", 10)))
            self.line_fps.setText(str(self.get_parameter("parameters.fps", 25)))
            self.line_start_frame.setText(str(self.get_parameter("parameters.start_frame", 0)))
            self.line_stop_frame.setText(str(self.get_parameter("parameters.stop_frame", 100)))
            self.line_w.setText(str(self.get_parameter("parameters.box_width", 70)))
            self.line_h.setText(str(self.get_parameter("parameters.box_height", 70)))
            self.line_delta.setText(str(self.get_parameter("parameters.delta", 5)))
            self.line_chop_sec.setText(str(self.get_parameter("parameters.chop_sec", 2)))
            self.checkbox_track.setChecked(self.get_parameter("parameters.tracking", True))
            self.checkbox_compare_first.setChecked(self.get_parameter("parameters.compare_to_first", False))
            self.checkbox_filter.setChecked(self.get_parameter("parameters.filter", True))
            self.checkbox_windowing.setChecked(self.get_parameter("parameters.windowing", True))
            self.checkbox_contrast.setChecked(self.get_parameter("parameters.contrast", True))
            self.checkbox_export.setChecked(self.get_parameter("parameters.export", False))
            self.checkbox_matlab.setChecked(self.get_parameter("parameters.matlab", False))

            self.view_position.setChecked(self.get_parameter("actions.position", False))
            self.view_position_x.setChecked(self.get_parameter("actions.position_x", False))
            self.view_position_y.setChecked(self.get_parameter("actions.position_y", False))
            self.view_position_all_on_one.setChecked(self.get_parameter("actions.position_all_on_one", True))
            self.view_phase.setChecked(self.get_parameter("actions.phase", False))
            self.view_violin.setChecked(self.get_parameter("actions.violin", False))
            self.view_violin_chop.setChecked(self.get_parameter("actions.violin_chop", False))
            self.view_violin_all_on_one.setChecked(self.get_parameter("actions.violin_all_on_one", True))
            self.view_step_length.setChecked(self.get_parameter("actions.step_length", False))
            self.view_experimental.setChecked(self.get_parameter("actions.experimental", False))

            print("Parameters loaded.")

    def save_parameters(self):
        self.saved_rectangles = {}  # Ensure moved boxes are saved with the updated coordinates

        i = 0
        for rectangle in self.rectangles:
            self.saved_rectangles[str(i)] = {
                "number": i,
                "x0": rectangle.pos()[0],
                "y0": rectangle.pos()[1],
                "width": rectangle.size()[0],
                "height": rectangle.size()[1]
            }

            i += 1

        self.json_data["boxes"][self.id] = self.saved_rectangles

        self.json_data = {
            "last_file": self.file_name,
            "parameters": {
                "pixel_size": float(self.line_pixel_size.text()),
                "magnification": int(self.line_magn.text()),
                "sub_pixel": int(self.line_sub_pixel.text()),
                "fps": int(self.line_fps.text()),
                "start_frame": int(self.line_start_frame.text()),
                "stop_frame": int(self.line_stop_frame.text()),
                "box_width": int(self.line_w.text()),
                "box_height": int(self.line_h.text()),
                "delta": int(self.line_delta.text()),
                "chop_sec": int(self.line_chop_sec.text()),
                "tracking": self.checkbox_track.isChecked(),
                "compare_to_first": self.checkbox_compare_first.isChecked(),
                "filter": self.checkbox_filter.isChecked(),
                "windowing": self.checkbox_windowing.isChecked(),
                "contrast": self.checkbox_contrast.isChecked(),
                "export": self.checkbox_export.isChecked(),
                "matlab": self.checkbox_matlab.isChecked()
            },
            "actions": {
                "position": self.view_position.isChecked(),
                "position_x": self.view_position_x.isChecked(),
                "position_y": self.view_position_y.isChecked(),
                "position_all_on_one": self.view_position_all_on_one.isChecked(),
                "phase": self.view_phase.isChecked(),
                "violin": self.view_violin.isChecked(),
                "violin_chop": self.view_violin_chop.isChecked(),
                "violin_all_on_one": self.view_violin_all_on_one.isChecked(),
                "step_length": self.view_step_length.isChecked(),
                "experimental": self.view_experimental.isChecked()
            },
            "boxes": self.json_data["boxes"]
        }

        with open(os.path.join(dirname, "settings.json"), "w") as json_file:
            json.dump(self.json_data, json_file, indent=4)

            print("Parameters saved.")

    def start_analysis(self):
        self.stop_analysis()  # ensure no analysis is already running
        # TODO: return if an analysis is already running instead of restarting a new analysis

        if self.video_data is None:  # no video loaded, return gracefully
            return

        if self.solver is not None:  # remove the arrows
            self.solver.clear_annotations()

        self.set_plot_options()
        self.save_parameters()

        self.output_basepath = utils.ensure_directory(self.file_name, "results")

        if self.checkbox_export.isChecked():
            write_target = utils.ensure_directory(self.file_name, "exports")
        else:
            write_target = None

        print(f"(Re)loading solver module...")
        import solver
        importlib.reload(solver)

        print(f"Tracking enabled: {self.checkbox_track.isChecked()}")
        self.solver = solver.Solver(
            video_data=self.video_data,
            rectangles=self.rectangles,
            fps=float(self.line_fps.text()),
            resolution=float(self.line_pixel_size.text()),
            upsample_factor=int(self.line_sub_pixel.text()),
            start_frame=int(self.line_start_frame.text()),
            stop_frame=int(self.line_stop_frame.text()),
            delta=int(self.line_delta.text()),
            track=self.checkbox_track.isChecked(),
            compare_first=self.checkbox_compare_first.isChecked(),
            filtering=self.checkbox_filter.isChecked(),
            windowing=self.checkbox_windowing.isChecked(),
            contrast=self.checkbox_contrast.isChecked(),
            matlab=self.checkbox_matlab.isChecked(),
            exports_folder=write_target,
            results_folder=self.output_basepath
        )

        pg.exporters.ImageExporter(self.pg_image_item.scene()).export(f"{self.output_basepath}_overview.png")

        # combined = skimage.color.gray2rgb(self.video_data.get_frame(int(self.line_start_frame.text()))).copy()
        # for rectangle in self.rectangles:
        #     position = rectangle.pos()
        #     top_left = (int(position.x()), int(position.y() + rectangle.size()[1]))
        #     bottom_right = (int(position.x() + rectangle.size()[0]), int(position.y()))
        #
        #     combined = cv2.rectangle(combined, top_left, bottom_right, (255, 0, 0), 1)
        #
        # skimage.io.imsave(f"{self.output_basepath}_raw.png", combined)
        skimage.io.imsave(f"{self.output_basepath}_raw.png", skimage.color.rgb2gray(self.video_data.get_frame(int(self.line_start_frame.text()))))

        self.solver.progress_signal.connect(self.update_progress)
        self.solver.arrow_signal.connect(self.update_arrow)
        self.solver.rectangle_signal.connect(self.update_rectangle)
        self.solver.start()

    def stop_analysis(self):
        print("Analysis stopped.")

        if self.solver is not None:
            self.solver.stop()

    def update_progress(self, progress, current_i, frame_n):
        if self.solver is not None:
            current_frame = current_i - int(self.line_start_frame.text())
            last_frame = int(self.line_stop_frame.text()) - int(self.line_start_frame.text())

            for j in range(len(self.rectangles)):
                item = self.rectangles_list.item(j)
                item.setText(f"{j} - {self.solver.progress}% (frame {current_frame}/{last_frame})")

            if progress == 100 or current_frame > 0:
                self.pg_image_item.setImage(frame_n)

            if progress == 100:
                if args.show_results:
                    self.show_results()

                if args.export_results:
                    self.export_results()

                if args.quit:
                    app.quit()

    def update_arrow(self, parameters):
        arrow = parameters["arrow"]
        if arrow is not None:
            arrow.setStyle(angle=parameters["angle"], headLen=parameters["headLen"], tailLen=parameters["tailLen"], pxMode=False)
            arrow.setPos(*parameters["pos"])
        else:
            arrow = pg.ArrowItem(parent=parameters["parent"], pos=parameters["pos"], angle=parameters["angle"], headLen=parameters["headLen"],
                                 tailLen=parameters["tailLen"], pxMode=False)

            self.solver.set_arrow(parameters["j"], arrow)

    def update_rectangle(self, j, dx, dy):
        pos = self.rectangles[j].pos()
        self.rectangles[j].setPos(pos.x() + dx, pos.y() + dy)

    def show_results(self):
        if self.solver is None or self.solver.progress < 100:
            return

        self.set_plot_options()
        self.save_parameters()  # only save parameters if there are plots to open

        self.opened_plots = utils.plot_results(
            pixels=self.solver.pixels,
            mean=self.solver.mean,
            shift_x=self.solver.shift_x,
            shift_y=self.solver.shift_y,
            shift_p=self.solver.shift_p,
            shift_x_y_error=self.solver.shift_x_y_error,
            box_shift=self.solver.box_shift,
            fps=self.solver.fps,
            res=self.solver.resolution,
            input_path=self.file_name,
            output_basepath=self.output_basepath,
            plots_dict=self.plots_dict,
            rectangles=self.rectangles,
            chop_duration=float(self.line_chop_sec.text()),
            start_frame=self.solver.start_frame
        )

        print(f"{len(self.opened_plots)} plots shown.")

    def reset_boxes(self):
        self.load_and_show_file(load_file=False)  # reloading the file resets everything

    def export_results(self):
        if self.solver is not None:
            utils.export_results(
                pixels=self.solver.pixels,
                mean=self.solver.mean,
                shift_x=self.solver.shift_x,
                shift_y=self.solver.shift_y,
                shift_p=self.solver.shift_p,
                shift_x_y_error=self.solver.shift_x_y_error,
                box_shift=self.solver.box_shift,
                fps=self.solver.fps,
                res=self.solver.resolution,
                output_basepath=self.output_basepath,
                rectangles=self.rectangles,
                start_frame=self.solver.start_frame
            )

        print("Files exported.")


if __name__ == "__main__":
    print(f"Python interpreter: {os.path.dirname(sys.executable)}, version: {sys.version}.")
    print(f"HDF5Plugin: {hdf5plugin.version}")

    sys.stdout.flush()

    parser = argparse.ArgumentParser(description="Nanomotion analysis software.")
    parser.add_argument("-o", "--open", help="File to open.", default=None)
    parser.add_argument("-a", "--autostart", help="Automatically start the analysis.", action="store_true")
    parser.add_argument("-r", "--show_results", help="Show results after the analysis.", action="store_true")
    parser.add_argument("-x", "--export_results", help="Export results after the analysis.", action="store_true")
    parser.add_argument("-q", "--quit", help="Quit after the analysis.", action="store_true")

    args = parser.parse_args()

    app = QApplication(sys.argv)

    menu = Main()
    menu.show()

    sys.exit(app.exec_())
