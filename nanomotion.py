import argparse
import hashlib
import json
import os
import platform
import sys

import PyQt5
import cv2
import h5py
import hdf5plugin
import numpy as np
import pims
import pyqtgraph
import pyqtgraph as pg
import pyqtgraph.exporters
import skimage.color
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.uic import loadUiType

import utils
from solver import Solver
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

            self.pg_image_item.setImage(skimage.color.rgb2gray(display))
            # self.pg_image_item.setImage(display)

            print("Shown: %s." % display.dtype)
        except Exception as e:
            print(e)
            self.pg_image_item.setImage(skimage.color.rgb2gray(self.video_data.get_frame(0)))
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

        print(f"Rectangle: {rectangle}, selected: {self.selected}")

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

    def load_parameters(self):
        with open(os.path.join(dirname, "settings.json"), "r") as json_file:
            self.json_data = json.load(json_file)

            if "last_file" in self.json_data:
                self.file_name = self.json_data["last_file"]

            self.line_pix_size.setText(str(self.json_data["parameters"]["pixel_size"]))
            self.line_magn.setText(str(self.json_data["parameters"]["magnification"]))
            self.line_sub_pix.setText(str(self.json_data["parameters"]["sub_pixel"]))
            self.line_fps.setText(str(self.json_data["parameters"]["fps"]))
            self.line_start_frame.setText(str(self.json_data["parameters"]["start_frame"]))
            self.line_stop_frame.setText(str(self.json_data["parameters"]["stop_frame"]))
            self.line_w.setText(str(self.json_data["parameters"]["box_width"]))
            self.line_h.setText(str(self.json_data["parameters"]["box_height"]))
            self.line_delta.setText(str(self.json_data["parameters"]["delta"]))
            self.line_chop_sec.setText(str(self.json_data["parameters"]["chop_sec"]))
            self.checkBox_track.setChecked(self.json_data["parameters"]["tracking"])
            self.checkBox_compare_first.setChecked(self.json_data["parameters"]["compare_to_first"])
            self.checkBox_filter.setChecked(self.json_data["parameters"]["filter"])
            self.checkBox_windowing.setChecked(self.json_data["parameters"]["windowing"])
            self.checkBox_export.setChecked(self.json_data["parameters"]["export"])
            self.checkBox_matlab.setChecked(self.json_data["parameters"]["matlab"])

            self.view_position.setChecked(self.json_data["actions"]["position"])
            self.view_position_x.setChecked(self.json_data["actions"]["position_x"])
            self.view_position_y.setChecked(self.json_data["actions"]["position_y"])
            self.view_position_all_on_one.setChecked(self.json_data["actions"]["position_all_on_one"])
            self.view_phase.setChecked(self.json_data["actions"]["phase"])
            self.view_violin.setChecked(self.json_data["actions"]["violin"])
            self.view_violin_chop.setChecked(self.json_data["actions"]["violin_chop"])
            self.view_violin_all_on_one.setChecked(self.json_data["actions"]["violin_all_on_one"])
            self.view_step_length.setChecked(self.json_data["actions"]["step_length"])
            self.view_experimental.setChecked(self.json_data["actions"]["experimental"])

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
                "pixel_size": float(self.line_pix_size.text()),
                "magnification": int(self.line_magn.text()),
                "sub_pixel": int(self.line_sub_pix.text()),
                "fps": int(self.line_fps.text()),
                "start_frame": int(self.line_start_frame.text()),
                "stop_frame": int(self.line_stop_frame.text()),
                "box_width": int(self.line_w.text()),
                "box_height": int(self.line_h.text()),
                "delta": int(self.line_delta.text()),
                "chop_sec": int(self.line_chop_sec.text()),
                "tracking": self.checkBox_track.isChecked(),
                "compare_to_first": self.checkBox_compare_first.isChecked(),
                "filter": self.checkBox_filter.isChecked(),
                "windowing": self.checkBox_windowing.isChecked(),
                "export": self.checkBox_export.isChecked(),
                "matlab": self.checkBox_matlab.isChecked()
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

        if self.checkBox_export.isChecked():
            write_target = utils.ensure_directory(self.file_name, "exports")
        else:
            write_target = None

        print("Tracking: %s." % (self.checkBox_track.isChecked()))
        self.solver = Solver(
            video_data=self.video_data,
            fps=float(self.line_fps.text()),
            rectangles=self.rectangles,
            upsample_factor=int(self.line_sub_pix.text()),
            stop_frame=int(self.line_stop_frame.text()),
            start_frame=int(self.line_start_frame.text()),
            res=float(self.line_pix_size.text()),
            track=self.checkBox_track.isChecked(),
            compare_first=self.checkBox_compare_first.isChecked(),
            delta=int(self.line_delta.text()),
            filter=self.checkBox_filter.isChecked(),
            windowing=self.checkBox_windowing.isChecked(),
            matlab=self.checkBox_matlab.isChecked(),
            write_target=write_target
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
            res=self.solver.res,
            input_path=self.file_name,
            output_basepath=self.output_basepath,
            plots_dict=self.plots_dict,
            rectangles=self.rectangles,
            chop_duration=float(self.line_chop_sec.text()),
            start_frame=self.solver.start_frame)

        print("%d plots shown." % (len(self.opened_plots)))

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
                res=self.solver.res,
                output_basepath=self.output_basepath,
                rectangles=self.rectangles,
                start_frame=self.solver.start_frame)

        print("Files exported.")


if __name__ == "__main__":
    print(f"Python interpreter: {os.path.dirname(sys.executable)}, version: {sys.version}.")
    print(f"HDF5Plugin: {hdf5plugin.version}")

    sys.stdout.flush()

    parser = argparse.ArgumentParser(description="Nanomotion software.")
    parser.add_argument("-o", "--open", help="File to open.", default=None)
    parser.add_argument("-a", "--autostart", help="Start the analysis.", action="store_true")
    parser.add_argument("-r", "--show_results", help="Show the results after the analysis.", action="store_true")
    parser.add_argument("-x", "--export_results", help="Export  the results after the analysis.", action="store_true")
    parser.add_argument("-q", "--quit", help="Quit after the analysis.", action="store_true")

    args = parser.parse_args()

    app = QApplication(sys.argv)

    menu = Main()
    menu.show()

    sys.exit(app.exec_())
