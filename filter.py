import numpy as np
import skimage.color
import skimage.exposure
import skimage.restoration


def filter_image(image: np.ndarray, contrast=False) -> np.ndarray:
    image = skimage.exposure.rescale_intensity(image * 1.0, in_range=(0, 255))  # ensure image is a float from -1 to 1

    if contrast:
        image = skimage.exposure.equalize_adapthist(image)

        image = skimage.restoration.denoise_tv_bregman(image, weight=15)

    # parameters = {
    #     "weight": np.arange(1, 20, 1),
    #     "max_num_iter": np.arange(100, 200, 10)
    # }
    # denoising_function, tested = skimage.restoration.calibrate_denoiser(image, skimage.restoration.denoise_tv_bregman, denoise_parameters=parameters,
    #                                                                     extra_output=True)
    #
    # print(tested)

    return image
