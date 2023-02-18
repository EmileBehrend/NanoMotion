import concurrent.futures
import io
import os

import cv2
import numpy as np
import scipy as sp
import skimage.color
import skimage.filters
import skimage.registration

import matplotlib
import matplotlib.patches
from PyQt5.QtCore import QThread, pyqtSignal


class Solver(QThread):
    progressChanged = pyqtSignal(int, int, object)

    def __init__(self, videodata, fps, res, box_dict, start_frame, stop_frame, upsample_factor,
                 track, compare_first, filter, windowing, matlab, figure, write_target):
        QThread.__init__(self)
        self.videodata = videodata  # store access to the video file to iterate over the frames
        self.figure = figure  # store the figure to draw displacement arrows

        self.fps = fps  # frames per seconds
        self.res = res  # size of one pixel (um / pixel)
        self.upsample_factor = upsample_factor
        self.track = track
        self.compare_first = compare_first
        self.filter = filter
        self.windowing = windowing
        self.matlab = matlab
        self.matlab_engine = None

        self.start_frame = start_frame
        self.stop_frame = stop_frame
        self.box_dict = box_dict

        self.centers_dict = [None for _ in self.box_dict]
        self.arrows_dict = [None for _ in self.box_dict]

        self.go_on = True

        self.row_min = []
        self.row_max = []
        self.col_min = []
        self.col_max = []

        self.pixels = [[None for _ in range(self.start_frame, self.stop_frame + 1)] for _ in self.box_dict]
        self.mean = [[None for _ in range(self.start_frame, self.stop_frame + 1)] for _ in self.box_dict]
        self.shift_x = [[None for _ in range(self.start_frame, self.stop_frame + 1)] for _ in self.box_dict]
        self.shift_y = [[None for _ in range(self.start_frame, self.stop_frame + 1)] for _ in self.box_dict]
        self.shift_p = [[None for _ in range(self.start_frame, self.stop_frame + 1)] for _ in self.box_dict]
        self.shift_x_y_error = [[None for _ in range(self.start_frame, self.stop_frame + 1)] for _ in self.box_dict]
        self.box_shift = [[[0, 0] for _ in range(self.start_frame, self.stop_frame + 1)] for _ in self.box_dict]
        self.cumulated_shift = [[0, 0] for _ in self.box_dict]

        self.frame_n = None
        self.progress = 0
        self.current_i = -1

        self.write_target = write_target

        self.debug_frames = []

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
        self.go_on = False
        print("Solver thread has been flagged to stop.")

    def clear_annotations(self):
        self.centers_dict.clear()

        for arrow in self.arrows_dict:
            if arrow is not None:
                # arrow.remove()
                self.figure.axes[0].patches.remove(arrow)

        self.arrows_dict.clear()

        self.figure.canvas.draw()

    def _close_to_zero(self, value):
        return int(value)  # integer casting truncates the number (1.2 -> 1, -1.2 -> -1)

    def _draw_arrow(self, j, dx, dy):
        ax = self.figure.add_subplot(111)

        current = self.arrows_dict[j]
        if current is not None:
            current.remove()

        arrow = matplotlib.patches.FancyArrow(self.centers_dict[j][0], self.centers_dict[j][1], dx, dy, width=2, head_length=1, head_width=4)

        self.arrows_dict[j] = ax.add_patch(arrow)

    def _prepare_image(self, image):
        image = skimage.color.rgb2gray(image)

        return image

    def _get_image_subset(self, image, j):
        return image[self.row_min[j]:self.row_max[j], self.col_min[j]:self.col_max[j]]

    def _filter_image_subset(self, image):
        pixels = 0

        if self.filter and len(self.debug_frames) == 0:
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
            shift, error, phase = skimage.registration.phase_cross_correlation(base_ft, current_ft, space="fourier", upsample_factor=upsample_factor)
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
        print("self.go_on is set to %s." % (self.go_on))

        images = []  # last image subimages array

        frame_first = self._prepare_image(self.videodata.get_frame(0))  # store the first subimages for later comparison

        for j in range(len(self.box_dict)):  # j iterates over all boxes
            image_first, pixels = self._filter_image_subset(self._get_image_subset(frame_first, j))
            images.append(image_first)

            print("Box %d (%d, %d, %d, %d)." % (j, self.row_min[j], self.row_max[j], self.col_min[j], self.col_max[j]))

            self.pixels[j][0] = pixels
            self.mean[j][0] = np.mean(image_first)
            self.shift_x[j][0] = 0
            self.shift_y[j][0] = 0
            self.shift_p[j][0] = 0
            self.shift_x_y_error[j][0] = 0
            self.box_shift[j][0] = [0, 0]

            self.centers_dict[j] = [
                int(self.box_dict[j].x_rect) + self.box_dict[j].rect._width / 2,
                int(self.box_dict[j].y_rect) + self.box_dict[j].rect._height / 2
            ]

            self._draw_arrow(j, 0, 0)

        length = self.stop_frame - self.start_frame
        progress_pivot = 0 - 5

        queue = {}
        for i in range(self.start_frame + 1, self.stop_frame + 1):  # i iterates over all frames
            if not self.go_on:  # condition checked to be able to stop the thread
                return

            self.current_i = i

            offset = i - self.start_frame

            if i not in queue:
                self.frame_n = self._prepare_image(self.videodata.get_frame(i))
            else:
                self.frame_n = self._prepare_image(queue[i].result())

            queue[i] = None

            if i < self.stop_frame:  # pooling the next image helps when analyzing a low number of cells
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    queue[i + 1] = executor.submit(self.videodata.get_frame, i + 1)

                    # TODO: properly implement multi-threaded reads
                    # for r in range(i + 1, i + min(5, self.stop_frame - i + 1)):  # pool next 5 frames
                    #     if r in queue:  # already pooled
                    #         continue
                    #
                    #     # print("Pooling frame: %d." % r)
                    #     queue[r] = executor.submit(self.videodata.get_frame, r)

            if self.write_target is not None:
                colored_frame_n = skimage.color.gray2rgba(self.frame_n)  # rgb (, 3) with alpha channel (, 4) because matplotlib.cm returns one (, 4)

            parameters = [None for _ in range(len(self.box_dict))]
            for j in range(len(self.box_dict)):  # j iterates over all boxes
                # Propagate previous box shifts
                self.box_shift[j][offset][0] = self.box_shift[j][offset - 1][0]
                self.box_shift[j][offset][1] = self.box_shift[j][offset - 1][1]

                # Shift before analysis (for the next frame)
                to_shift = [0, 0]
                if self.track and (abs(self.cumulated_shift[j][0]) >= 1.2 or abs(self.cumulated_shift[j][1]) >= 1.2):  # arbitrary value (1.2)
                    to_shift = [
                        self._close_to_zero(self.cumulated_shift[j][0]),
                        self._close_to_zero(self.cumulated_shift[j][1])
                    ]

                    print("Box %d - to shift: (%f, %f), cumulated shift: (%f, %f)." % (
                        j, to_shift[0], to_shift[1], self.cumulated_shift[j][0], self.cumulated_shift[j][1]))

                    self.cumulated_shift[j][0] -= to_shift[0]
                    self.cumulated_shift[j][1] -= to_shift[1]

                    self.box_shift[j][offset][0] += to_shift[0]
                    self.box_shift[j][offset][1] += to_shift[1]

                    print("Box %d - shifted at frame %d (~%ds)." % (j, i, i / self.fps))

                    self.box_dict[j].x_rect += to_shift[0]
                    self.box_dict[j].y_rect += to_shift[1]
                    self._crop_coord(j)

                parameters[j] = [images[j], self._get_image_subset(self.frame_n, j), self.upsample_factor]

            results = self._run_threading(parameters)

            for j in range(len(self.box_dict)):
                image_n, pixels, shift, error, phase = results[j]
                self.pixels[j][offset] = pixels
                self.mean[j][offset] = np.mean(image_n)

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
                        shift[0] - self.shift_x[j][offset - 1],
                        shift[1] - self.shift_y[j][offset - 1],
                        phase - self.shift_p[j][offset - 1]
                    ]

                    # computed_error = error
                else:
                    relative_shift = [
                        shift[0],
                        shift[1],
                        phase
                    ]

                    # computed_error = error + self.shift_x_y_error[j][offset - 1]

                self.cumulated_shift[j][0] += relative_shift[0]
                self.cumulated_shift[j][1] += relative_shift[1]

                self.shift_x[j][offset] = self.shift_x[j][offset - 1] + relative_shift[0]
                self.shift_y[j][offset] = self.shift_y[j][offset - 1] + relative_shift[1]
                self.shift_p[j][offset] = self.shift_p[j][offset - 1] + relative_shift[2]

                self.shift_x_y_error[j][offset] = computed_error

                # TODO: phase difference
                # TODO: take into account rotation (https://scikit-image.org/docs/stable/auto_examples/registration/plot_register_rotation.html).

                self._draw_arrow(j, self.shift_x[j][offset] + self.box_shift[j][offset][0], self.shift_y[j][offset] + self.box_shift[j][offset][1])

                # if j == 1:
                #     print("Box %d - raw shift: (%f, %f), relative shift: (%f, %f), cumulated shift: (%f, %f), error: %f."
                #           % (j, shift[0], shift[1], relative_shift[0], relative_shift[1], self.cumulated_shift[j][0], self.cumulated_shift[j][1], error))

                if self.write_target is not None:
                    colored_frame_n = self._combine_subset(colored_frame_n, j, image_n)

                if not self.compare_first:  # store the current image to be compared later
                    images[j] = image_n

                if i in self.debug_frames:
                    print("Box %d, frame %d." % (j, i))
                    print("Previous shift: %f, %f." % (self.shift_x[j][offset - 1], self.shift_y[j][offset - 1]))
                    print("Current shift: %f, %f." % (shift[0], shift[1]))
                    print("Relative shift: %f, %f." % (relative_shift[0], relative_shift[1]))

                    os.makedirs("./debug/", exist_ok=True)

                    # Previous image (i - 1 or 0): images[j]
                    if self.compare_first:
                        cv2.imwrite("./debug/box_%d-0.png" % (j), images[j])

                        previous = self.videodata.get_frame(i - 1)[self.row_min[j]:self.row_max[j], self.col_min[j]:self.col_max[j]]
                        cv2.imwrite(("./debug/box_%d-%d_p.png" % (j, i - 1)), previous)
                    else:
                        cv2.imwrite(("./debug/box_%d-%d_p.png" % (j, i - 1)), images[j])

                    # Current image (i): parameters[j][0]
                    cv2.imwrite(("./debug/box_%d-%d.png" % (j, i)), image_n)

            if self.write_target is not None:
                cv2.imwrite("%s_image_%d.png" % (self.write_target, i), colored_frame_n * 255)

            self.progress = int(((i - self.start_frame) / length) * 100)
            if self.progress > progress_pivot + 4:
                print("%d%% (frame: %d/%d, real frame: %d)." % (self.progress, offset, self.stop_frame - self.start_frame, i))
                progress_pivot = self.progress

        # Post-process data (invert y-dimension)
        for j in range(len(self.box_dict)):
            inverted_shift_y = [-shift_y for shift_y in self.shift_y[j]]
            self.shift_y[j] = inverted_shift_y

            inverted_box_shift = [[entry[0], -entry[1]] for entry in self.box_shift[j]]
            self.box_shift[j] = inverted_box_shift

    def _crop_coord(self, which=-1):
        if which == -1:
            self.row_min.clear()
            self.row_max.clear()
            self.col_min.clear()
            self.col_max.clear()

            for i in range(len(self.box_dict)):
                self.row_min.append(int(self.box_dict[i].y_rect))
                self.row_max.append(int(self.box_dict[i].y_rect) + self.box_dict[i].rect._height)
                self.col_min.append(int(self.box_dict[i].x_rect))
                self.col_max.append(int(self.box_dict[i].x_rect) + self.box_dict[i].rect._width)
        else:
            i = which

            self.row_min[i] = int(self.box_dict[i].y_rect)
            self.row_max[i] = int(self.box_dict[i].y_rect) + self.box_dict[i].rect._height
            self.col_min[i] = int(self.box_dict[i].x_rect)
            self.col_max[i] = int(self.box_dict[i].x_rect) + self.box_dict[i].rect._width
