from __future__ import annotations

import h5py
import numpy as np
from pims import Video, ImageSequence


class VideoSequence:
    def get_frame(self, number: int) -> np.ndarray:
        raise NotImplementedError("Not implemented.")

    def next(self) -> (np.ndarray | None, int | None):
        raise NotImplementedError("Not implemented.")

    def reset_iterator(self):
        raise NotImplementedError("Not implemented.")


class PimsSequence(VideoSequence):
    def __init__(self, pims_object: Video | ImageSequence):
        VideoSequence.__init__(self)

        self.pims_object = pims_object

        self.length = len(self.pims_object)
        self.number = 0

    def get_frame(self, number: int) -> np.ndarray:
        return self.pims_object.get_frame(number)

    def next(self) -> (np.ndarray | None, int | None):
        if self.number + 2 >= self.length:
            return None, None
        else:
            frame = self.pims_object.get_frame(self.number)

            frame_number = self.number
            self.number += 1

            return frame, frame_number

    def reset_iterator(self):
        self.number = 0


class H5Sequence(VideoSequence):
    def __init__(self, h5_file: h5py.File):
        VideoSequence.__init__(self)

        self.h5_dataset: h5py.Dataset = h5_file["stack"]

        self.length = len(self.h5_dataset)
        self.number = 0

        self.chunks = None
        self.chunks_loaded_data = None
        self.chunks_loaded_index = 0
        self.chunks_last_loaded = 0

    def get_frame(self, number: int) -> np.ndarray:
        return self.h5_dataset[number]

    def next(self) -> (np.ndarray | None, int | None):
        if self.number + 2 >= self.length:
            return None, None
        else:
            if self.chunks is None:
                self.chunks = self.h5_dataset.iter_chunks()

            if self.chunks_loaded_data is None or self.number >= self.chunks_last_loaded:
                self.chunks_loaded_data = self.h5_dataset[next(self.chunks)]
                self.chunks_loaded_index = 0

                self.chunks_last_loaded += len(self.chunks_loaded_data)
            else:
                self.chunks_loaded_index += 1

        frame_number = self.number
        self.number += 1

        return self.chunks_loaded_data[self.chunks_loaded_index], frame_number

    def reset_iterator(self):
        self.number = 0

        self.chunks = None
        self.chunks_loaded_data = None
        self.chunks_loaded_index = 0
        self.chunks_last_loaded = 0
