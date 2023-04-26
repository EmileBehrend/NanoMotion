import concurrent.futures
import importlib
import io
import math
import os
import threading

import cv2
import matplotlib
import matplotlib.patches
import numpy as np
import scipy as sp
import skimage.color
import skimage.filters
import skimage.registration
import skimage.restoration
from PyQt5.QtCore import QThread, pyqtSignal

from video_backend import VideoSequence


class Solver(QThread):
    progress_signal = pyqtSignal(int, int, object)
    arrow_signal = pyqtSignal(object)
    rectangle_signal = pyqtSignal(int, int, int)
    mutex = threading.Lock()

    def __init__(self, video_data, rectangles, fps, resolution, upsample_factor, start_frame, stop_frame, track, compare_first, delta, filtering, windowing,
                 contrast,
                 matlab, write_target):
        QThread.__init__(self)
        self.video_data: VideoSequence = video_data  # store access to the video file to iterate over the frames

        self.fps = fps  # frames per seconds
        self.resolution = resolution  # size of one pixel (um / pixel)
        self.upsample_factor = upsample_factor
        self.delta = delta
        self.track = track
        self.compare_first = compare_first
        self.filtering = filtering
        self.windowing = windowing
        self.contrast = contrast
        self.matlab = matlab
        self.matlab_engine = None

        self.start_frame = start_frame
        self.stop_frame = stop_frame

        self.rectangles = rectangles
        self.local_rectangles = []

        for rectangle in self.rectangles:
            pos = rectangle.pos()
            width, height = rectangle.size()

            self.local_rectangles.append({"x": pos.x(), "y": pos.y(), "width": width, "height": height})

        self.centers = [None for _ in self.local_rectangles]

        with self.mutex:
            self.arrows = [None for _ in self.local_rectangles]

            self.go_on = True

        self.row_min = []
        self.row_max = []
        self.col_min = []
        self.col_max = []

        self.pixels = [[None for _ in range(self.start_frame, self.stop_frame + 1)] for _ in self.local_rectangles]
        self.mean = [[None for _ in range(self.start_frame, self.stop_frame + 1)] for _ in self.local_rectangles]
        self.shift_x = [[None for _ in range(self.start_frame, self.stop_frame + 1)] for _ in self.local_rectangles]
        self.shift_y = [[None for _ in range(self.start_frame, self.stop_frame + 1)] for _ in self.local_rectangles]
        self.shift_p = [[None for _ in range(self.start_frame, self.stop_frame + 1)] for _ in self.local_rectangles]
        self.shift_x_y_error = [[None for _ in range(self.start_frame, self.stop_frame + 1)] for _ in self.local_rectangles]
        self.box_shift = [[[0, 0] for _ in range(self.start_frame, self.stop_frame + 1)] for _ in self.local_rectangles]
        self.cumulated_shift = [[0, 0] for _ in self.local_rectangles]

        self.frame_n = None
        self.progress = 0
        self.current_i = -1

        self.write_target = write_target

        self.debug_frames = []

    def set_arrow(self, j, arrow):
        with self.mutex:
            self.arrows[j] = arrow

    def run(self):
        try:
            if self.matlab:
                import matlab
                import matlab.engine

                self.matlab_engine = matlab.engine.start_matlab()
            self._crop_coord()
            self._compute_phase_corr()
        except UserWarning:
            self.stop()

        if self.matlab_engine is not None:
            self.matlab_engine.quit()

    def stop(self):
        with self.mutex:
            self.go_on = False
        print("Solver thread has been flagged to stop.")

    def clear_annotations(self):
        self.centers.clear()

        with self.mutex:
            for arrow in self.arrows:
                arrow.setParentItem(None)

            self.arrows.clear()

    def _close_to_zero(self, value):
        return int(value)  # integer casting truncates the number (1.2 -> 1, -1.2 -> -1)

    def _draw_arrow(self, j, i, dx, dy):
        hypotenuse = np.sqrt(dx ** 2 + dy ** 2)

        angle = math.degrees(math.acos(dx / hypotenuse))

        # print(f"Hypotenuse: {hypotenuse}, angle: {angle}")
        tail_length = hypotenuse  # TODO: customize this
        head_length = 1

        rectangle = self.local_rectangles[j]
        width = rectangle["width"]
        height = rectangle["height"]
        starting_pos = (width / 2 - self.box_shift[j][i][0], height / 2 - self.box_shift[j][i][1])

        with self.mutex:
            parameters = {
                "j": j,
                "arrow": self.arrows[j],
                "parent": self.rectangles[j],
                "pos": starting_pos,
                "angle": angle,
                "headLen": head_length,
                "tailLen": tail_length
            }

            self.arrow_signal.emit(parameters)

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        image = skimage.color.rgb2gray(image)

        import filter
        importlib.reload(filter)

        image = filter.filter_image(image, contrast=self.contrast)

        return image

    def _get_image_subset(self, image: np.ndarray, j: int) -> np.ndarray:
        return image[self.row_min[j]:self.row_max[j], self.col_min[j]:self.col_max[j]]

    def _filter_image_subset(self, image: np.ndarray) -> (np.ndarray, int):
        pixels = 0

        if self.filtering and len(self.debug_frames) == 0:
            image = skimage.filters.difference_of_gaussians(image, 0.5, 25)

        if self.windowing:
            # image = image * skimage.filters.window("hann", image.shape)
            image = image * skimage.filters.window("hamming", image.shape)

            # TODO: improve image segmentation
            """
            Canny segmentation
            """
            # edges = skimage.feature.canny(image)
            # segmented = sp.ndimage.binary_fill_holes(edges)

            """
            Multiotsu segmentation
            """
            # thresholds = skimage.filters.threshold_multiotsu(image, classes=4)
            # inner = (image > thresholds[0]) < thresholds[1]
            # outer = skimage.util.invert((image > thresholds[1]) < thresholds[2])
            #
            # dilated = skimage.morphology.binary_dilation(inner, skimage.morphology.disk(3))
            # filled = sp.ndimage.binary_fill_holes(dilated)
            # segmented = outer + filled

            # regions = np.digitize(image, bins=thresholds)
            #
            # image = regions

            """
            Li segmentation
            """
            threshold = skimage.filters.threshold_li(image)
            segmented = sp.ndimage.binary_fill_holes(image <= threshold)

            # image = segmented  # debug

            pixels = np.count_nonzero(segmented)
            # print(pixels)  # debug

        return image, pixels

    def _phase_cross_correlation_wrapper(self, base, current, upsample_factor):
        current, pixels = self._filter_image_subset(current)  # only filter 'current' image as 'base' was already filtered in the previous pass

        # reusing the Fourier Transform later doesn't lead to noticeable performance improvements but instead makes debugging impossible
        base_ft = np.fft.fft2(base)
        current_ft = np.fft.fft2(current)

        if self.matlab is False:
            # `disambiguate` to False produces better results
            # `normalization` must be set to None
            shift, error, phase = skimage.registration.phase_cross_correlation(base_ft, current_ft, upsample_factor=upsample_factor, space="fourier",
                                                                               disambiguate=False, normalization=None)
        else:
            import matlab
            import matlab.engine

            reference = matlab.double(base_ft.tolist(), is_complex=True)
            moved = matlab.double(current_ft.tolist(), is_complex=True)

            # output = [error, diffphase, net_row_shift, net_col_shift]
            output, Greg = self.matlab_engine.dftregistration(reference, moved, upsample_factor, nargout=2, stdout=io.StringIO())

            error, phase, row_shift, col_shift = output[0]
            shift = [row_shift, col_shift]

        return current, pixels, shift, error, phase

    def _run_threading(self, parameters):
        futures = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for entry in parameters:
                future = executor.submit(self._phase_cross_correlation_wrapper, entry[0], entry[1], entry[2])
                futures.append(future)

        return [future.result() for future in futures]

    def _combine_subset(self, colored_frame_n, j, image_n):
        # TODO: correct colormap range
        # colored_subset = matplotlib.cm.magma(image_n)

        mappable = matplotlib.cm.ScalarMappable(matplotlib.colors.Normalize(), matplotlib.cm.cividis)
        colored_subset = mappable.to_rgba(image_n)

        colored_frame_n[self.row_min[j]:self.row_max[j], self.col_min[j]:self.col_max[j]] = colored_subset

        return colored_frame_n

    def _compute_phase_corr(self):  # compute the cross-correlation for all frames for the selected polygon crop
        with self.mutex:
            print(f"self.go_on is set to {self.go_on}.")

        images = []  # last image subimages array

        delta_images = []
        if self.delta > 1:  # TODO: rework this
            for i in range(self.delta):
                frame = self._prepare_image(self.video_data.get_frame(self.start_frame + i))

                delta_j = []
                for j in range(len(self.local_rectangles)):  # j iterates over all boxes
                    frame_j, pixels = self._filter_image_subset(self._get_image_subset(frame, j))

                    delta_j.append(frame_j)

                    # TODO: calculate shifts between delta pixels
                    self.pixels[j][i] = pixels
                    self.mean[j][i] = np.mean(frame_j)
                    self.shift_x[j][i] = 0
                    self.shift_y[j][i] = 0
                    self.shift_p[j][i] = 0
                    self.shift_x_y_error[j][i] = 0
                    self.box_shift[j][0] = [0, 0]

                delta_images.append(delta_j)

        frame_first = self._prepare_image(self.video_data.get_frame(self.start_frame))  # store the starting subimages for later comparison

        for j in range(len(self.local_rectangles)):  # j iterates over all boxes
            image_first, pixels = self._filter_image_subset(self._get_image_subset(frame_first, j))

            images.append(image_first)

            # print("Box %d (%d, %d, %d, %d)." % (j, self.row_min[j], self.row_max[j], self.col_min[j], self.col_max[j]))

            rectangle = self.local_rectangles[j]
            x = rectangle["x"]
            y = rectangle["y"]
            width = rectangle["width"]
            height = rectangle["height"]

            self.centers[j] = [
                x + width / 2,
                y + height / 2
            ]

            self._draw_arrow(j, 0, 0, 0)

            self.pixels[j][0] = pixels
            self.mean[j][0] = np.mean(image_first)
            self.shift_x[j][0] = 0
            self.shift_y[j][0] = 0
            self.shift_p[j][0] = 0
            self.shift_x_y_error[j][0] = 0
            self.box_shift[j][0] = [0, 0]

        length = self.stop_frame - self.start_frame
        progress_pivot = 0 - 5

        queue = {}
        for i in range(self.start_frame + self.delta, self.stop_frame + 1):  # i iterates over all frames
            with self.mutex:
                if not self.go_on:  # condition checked to be able to stop the thread
                    return

            from_start = i - self.start_frame

            self.current_i = i

            read_frame, read_number = self.video_data.next()
            # print(f"Number: {read_number}, frame: {np.shape(read_frame)}")

            if read_number > i:
                self.video_data.reset_iterator()

                read_frame, read_number = self.video_data.next()

            while read_number < i:  # i is always >= 1 (because frame 0 is the first frame reference)
                read_frame, read_number = self.video_data.next()
                # print(f"Number: {read_number}, frame: {np.shape(read_frame)}")

            self.frame_n = self._prepare_image(read_frame)

            # if i not in queue:
            #     self.frame_n = self._prepare_image(self.video_data.get_frame(i))
            # else:
            #     self.frame_n = self._prepare_image(queue[i].result())
            #
            # queue[i] = None
            #
            # if i < self.stop_frame:  # pooling the next image helps when analyzing a low number of cells
            #     with concurrent.futures.ThreadPoolExecutor() as executor:
            #         queue[i + 1] = executor.submit(self.video_data.get_frame, i + 1)
            #
            #         # TODO: properly implement multi-threaded reads
            #         # for r in range(i + 1, i + min(5, self.stop_frame - i + 1)):  # pool next 5 frames
            #         #     if r in queue:  # already pooled
            #         #         continue
            #         #
            #         #     # print("Pooling frame: %d." % r)
            #         #     queue[r] = executor.submit(self.videodata.get_frame, r)

            if self.write_target is not None:
                colored_frame_n = skimage.color.gray2rgba(self.frame_n)  # rgb (, 3) with alpha channel (, 4) because matplotlib.cm returns one (, 4)
                # colored_frame_n = self.frame_n.copy()  # rgb (, 3) with alpha channel (, 4) because matplotlib.cm returns one (, 4)

            parameters = [None for _ in range(len(self.local_rectangles))]
            for j in range(len(self.local_rectangles)):  # j iterates over all boxes
                # Propagate previous box shifts
                self.box_shift[j][from_start][0] = self.box_shift[j][from_start - 1][0]
                self.box_shift[j][from_start][1] = self.box_shift[j][from_start - 1][1]

                # Shift before analysis (for the next frame)
                to_shift = [0, 0]
                if self.track and (abs(self.cumulated_shift[j][0]) >= 1.2 or abs(self.cumulated_shift[j][1]) >= 1.2):  # arbitrary value (1)
                    to_shift = [
                        self._close_to_zero(self.cumulated_shift[j][0]),
                        self._close_to_zero(self.cumulated_shift[j][1])
                    ]

                    print("Box %d - to shift: (%f, %f), cumulated shift: (%f, %f)." % (
                        j, to_shift[0], to_shift[1], self.cumulated_shift[j][0], self.cumulated_shift[j][1]))

                    self.cumulated_shift[j][0] -= to_shift[0]
                    self.cumulated_shift[j][1] -= to_shift[1]

                    self.box_shift[j][from_start][0] += to_shift[0]
                    self.box_shift[j][from_start][1] += to_shift[1]

                    print("Box %d - shifted at frame %d (~%ds)." % (j, i, i / self.fps))
                    rectangle = self.local_rectangles[j]
                    rectangle["x"] += to_shift[0]
                    rectangle["y"] += to_shift[1]

                    self.rectangle_signal.emit(j, to_shift[0], to_shift[1])

                    self._crop_coord(j)

                parameters[j] = [images[j], self._get_image_subset(self.frame_n, j), self.upsample_factor]

            results = self._run_threading(parameters)

            for j in range(len(self.local_rectangles)):
                image_n, pixels, shift, error, phase = results[j]
                self.pixels[j][from_start] = pixels
                self.mean[j][from_start] = np.mean(image_n)

                shift = list(shift)  # convert tuple to list
                shift[0], shift[1] = -shift[1], -shift[0]  # (-y, -x) → (x, y)

                # shift[0] is the x displacement computed by comparing the first (if self.compare_first is True) or
                # the previous image with the current image.
                # shift[1] is the y displacement.
                #
                # We need to store the absolute displacement [position(t)] as shift_x and shift_y.
                # We can later compute the displacement relative to the previous frame [delta_position(t)].
                #
                # If compare_first is True, the shift values represent the absolute displacement.
                # In the other case, the shift values represent the relative displacement.

                relative_shift = [0, 0]
                computed_error = 0
                # TODO: fix error computation
                if self.compare_first:
                    relative_shift = [
                        shift[0] - self.shift_x[j][from_start - 1],
                        shift[1] - self.shift_y[j][from_start - 1],
                        phase - self.shift_p[j][from_start - 1]
                    ]

                    # computed_error = error
                else:
                    if self.delta > 1:
                        # TODO: fix delta subtracting
                        # print(f"from_start: {from_start}, x: {self.shift_x[j]}")
                        #
                        # relative_shift = [
                        #     shift[0] + self.shift_x[j][from_start - self.delta] - self.shift_x[j][from_start - 1],
                        #     shift[1] + self.shift_y[j][from_start - self.delta] - self.shift_y[j][from_start - 1],
                        #     phase + self.shift_p[j][from_start - self.delta] - self.shift_p[j][from_start - 1]
                        # ]

                        relative_shift = [
                            shift[0],
                            shift[1],
                            phase
                        ]
                    else:
                        relative_shift = [
                            shift[0],
                            shift[1],
                            phase
                        ]

                        # computed_error = error + self.shift_x_y_error[j][offset - 1]

                self.cumulated_shift[j][0] += relative_shift[0]
                self.cumulated_shift[j][1] += relative_shift[1]

                # print(f"From start: {from_start}")
                # print(f"Shift x: {self.shift_x[j]}")

                self.shift_x[j][from_start] = self.shift_x[j][from_start - 1] + relative_shift[0]
                self.shift_y[j][from_start] = self.shift_y[j][from_start - 1] + relative_shift[1]
                self.shift_p[j][from_start] = self.shift_p[j][from_start - 1] + relative_shift[2]

                self.shift_x_y_error[j][from_start] = computed_error

                # TODO: phase difference
                # TODO: take into account rotation (https://scikit-image.org/docs/stable/auto_examples/registration/plot_register_rotation.html).

                self._draw_arrow(j, from_start, self.shift_x[j][from_start] + self.box_shift[j][from_start][0],
                                 self.shift_y[j][from_start] + self.box_shift[j][from_start][1])

                # if j == 1:
                #     print("Box %d - raw shift: (%f, %f), relative shift: (%f, %f), cumulated shift: (%f, %f), error: %f."
                #           % (j, shift[0], shift[1], relative_shift[0], relative_shift[1], self.cumulated_shift[j][0], self.cumulated_shift[j][1], error))

                if self.write_target is not None:
                    colored_frame_n = self._combine_subset(colored_frame_n, j, image_n)

                if not self.compare_first:
                    if self.delta > 1:
                        images[j] = delta_images[1][j]  # TODO: generalize delta (1 means previous image, higher means storing a list)

                        delta_images[0][j] = image_n  # first array will be moved to the last position later
                    else:  # store the current image to be compared later
                        images[j] = image_n

                if i in self.debug_frames:
                    print("Box %d, frame %d." % (j, i))
                    print("Previous shift: %f, %f." % (self.shift_x[j][from_start - 1], self.shift_y[j][from_start - 1]))
                    print("Current shift: %f, %f." % (shift[0], shift[1]))
                    print("Relative shift: %f, %f." % (relative_shift[0], relative_shift[1]))

                    os.makedirs("./debug/", exist_ok=True)

                    # Previous image (i - 1 or 0): images[j]
                    if self.compare_first:
                        cv2.imwrite("./debug/box_%d-first.png" % j, self._get_image_subset(self.video_data.get_frame(0), j))

                        cv2.imwrite("./debug/box_%d-%d_p.png" % (j, i - 1), self._get_image_subset(self.video_data.get_frame(i - 1), j))
                    else:
                        cv2.imwrite("./debug/box_%d-%d_p.png" % (j, i - 1), images[j])

                    # Current image (i): parameters[j][0]
                    cv2.imwrite(("./debug/box_%d-%d.png" % (j, i)), self._get_image_subset(self.video_data.get_frame(i), j))

            if self.delta > 1:
                delta_images.append(delta_images.pop(0))  # move first element (which was updated now) to the last position

            if self.write_target is not None:
                cv2.imwrite("%s_image_%d.png" % (self.write_target, i), colored_frame_n * 255)

            self.progress = int(((i - self.start_frame) / length) * 100)
            if self.progress > progress_pivot + 4:
                print("%d%% (frame: %d/%d, real frame: %d)." % (self.progress, from_start, self.stop_frame - self.start_frame, i))
                progress_pivot = self.progress

            self.progress_signal.emit(self.progress, self.current_i, self.frame_n)

        # Post-process data (invert y-dimension)
        for j in range(len(self.local_rectangles)):
            inverted_shift_y = [-shift_y for shift_y in self.shift_y[j]]
            self.shift_y[j] = inverted_shift_y

            inverted_box_shift = [[entry[0], -entry[1]] for entry in self.box_shift[j]]
            self.box_shift[j] = inverted_box_shift

    def _crop_coord(self, target=None):
        if target is None:
            self.row_min.clear()
            self.row_max.clear()
            self.col_min.clear()
            self.col_max.clear()

            for rectangle in self.local_rectangles:
                x = rectangle["x"]
                y = rectangle["y"]
                width = rectangle["width"]
                height = rectangle["height"]

                self.row_min.append(int(y))
                self.row_max.append(int(y + height))
                self.col_min.append(int(x))
                self.col_max.append(int(x + width))
        else:
            rectangle = self.local_rectangles[target]

            x = rectangle["x"]
            y = rectangle["y"]
            width = rectangle["width"]
            height = rectangle["height"]

            self.row_min[target] = int(y)
            self.row_max[target] = int(y + height)
            self.col_min[target] = int(x)
            self.col_max[target] = int(x + width)
