import cv2
import numpy as np
from typing import Union


class ImageModifiers:

    @staticmethod
    def apply_median_blur(image, kernel_size: int = 51):
        kernel_size = _round_to_nearest_odd_int(kernel_size)
        return cv2.medianBlur(image, kernel_size)

    @staticmethod
    def apply_gaussian_blur(image, kernel_size: int = 51):
        if kernel_size == 0:
            return image
        kernel_size = _round_to_nearest_odd_int(kernel_size)
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    @staticmethod
    def apply_gaussian_noise(image, mean: int = 0, standard_deviation: int = 5):
        noise = np.random.normal(mean, standard_deviation, image.shape).astype(np.uint8)
        return cv2.add(image, noise)

    @staticmethod
    def apply_salt_and_pepper_noise(image, noise_ratio: float = .1):
        # add salt noise (random white pixels)
        noisy_image = ImageModifiers._replace_random_pixels(image, value=255, replace_ratio=noise_ratio/2.)
        # add pepper noise (random black pixels)
        return ImageModifiers._replace_random_pixels(noisy_image, value=0, replace_ratio=noise_ratio/2.)

    @staticmethod
    def apply_hue_multiplication(image, hue_factor: float = 1.):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv_image[:, :, 0] = np.uint8(np.clip((hsv_image[:, :, 0] * hue_factor) % 360, 0, 360))
        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    @staticmethod
    def apply_hue_multiplication_to_area(
            image,
            area_centre_width: float = .5,
            area_centre_height: float = .5,
            area_width: float = 1.,
            area_height: float = 1.,
            mod_factor: float = 1.
    ):
        image_width = image.shape[1]
        image_height = image.shape[0]
        centre_width_pixel = int(area_centre_width * image_width)
        centre_height_pixel = int(area_centre_height * image_height)
        width_pixels = _round_to_nearest_even_int(image_width * area_width)
        height_pixels = _round_to_nearest_even_int(image_height * area_height)
        left = int(np.clip(centre_width_pixel - width_pixels / 2, 0, image_width))
        right = int(np.clip(centre_width_pixel + width_pixels / 2, 0, image_width))
        top = int(np.clip(centre_height_pixel + height_pixels / 2, 0, image_width))
        bottom = int(np.clip(centre_width_pixel - height_pixels / 2, 0, image_width))
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv_image[bottom:top, left:right, 0] = np.uint8(
            np.clip((hsv_image[bottom:top, left:right, 0] * mod_factor) % 360, 0, 360)
        )
        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    @staticmethod
    def apply_saturation_multiplication(image, saturation_factor: float = 1.):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv_image[:, :, 1] = np.uint8(np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255))
        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    @staticmethod
    def apply_rgb_multiplication(image, red_factor: float = 1., green_factor: float = 1., blue_factor: float = 1.):
        return np.uint8(
            np.stack(
                [
                    np.clip(image[:, :, 0] * red_factor, 0, 255),
                    np.clip(image[:, :, 1] * green_factor, 0, 255),
                    np.clip(image[:, :, 2] * blue_factor, 0, 255)
                ], axis=-1)
        )

    @staticmethod
    def apply_red_scaling(image, scale_factor: float = 0.):
        return ImageModifiers._scale_rgb_channel(image, scale_factor, 0)

    @staticmethod
    def apply_green_scaling(image, scale_factor: float = 0.):
        return ImageModifiers._scale_rgb_channel(image, scale_factor, 1)

    @staticmethod
    def apply_blue_scaling(image, scale_factor: float = 0.):
        return ImageModifiers._scale_rgb_channel(image, scale_factor, 2)

    @staticmethod
    def _scale_rgb_channel(image, scale_factor: float, channel: int):
        if scale_factor < 0.:
            modified_channel = image[:, :, channel] + scale_factor * image[:, :, channel]
        elif scale_factor > 0.:
            modified_channel = image[:, :, channel] + scale_factor * (255 - image[:, :, channel])
        else:
            return image
        arrays = []
        for i in [0, 1, 2]:
            if i == channel:
                arrays.append(np.clip(modified_channel, 0, 255))
            else:
                arrays.append(image[:, :, i])
        return np.uint8(np.stack(arrays, axis=-1))

    @staticmethod
    def apply_ghost_images(image, number_of_ghost_images: int = 5, max_shift: int = 75, alpha: float = .1):
        number_of_ghost_images = int(number_of_ghost_images)
        max_shift = int(max_shift)

        output = image.copy()

        # Generate spatially shifted versions and blend them
        for i in range(number_of_ghost_images):
            dx = np.random.randint(-max_shift, max_shift + 1)
            dy = np.random.randint(-max_shift, max_shift + 1)

            m = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted_image = cv2.warpAffine(
                image,
                m,
                (image.shape[1], image.shape[0])
            )
            output = cv2.addWeighted(output, 1, shifted_image, alpha, 0)
        return np.uint8(output)

    @staticmethod
    def apply_red_vlines(image, no_lines: int = 10, min_thickness: int = 1, max_thickness: int = 8):
        return ImageModifiers._apply_coloured_vlines(image, no_lines, min_thickness, max_thickness, 0)

    @staticmethod
    def apply_green_vlines(image, no_lines: int = 10, min_thickness: int = 1, max_thickness: int = 8):
        return ImageModifiers._apply_coloured_vlines(image, no_lines, min_thickness, max_thickness, 1)

    @staticmethod
    def apply_blue_vlines(image, no_lines: int = 10, min_thickness: int = 1, max_thickness: int = 8):
        return ImageModifiers._apply_coloured_vlines(image, no_lines, min_thickness, max_thickness, 2)

    @staticmethod
    def apply_random_coloured_hlines(image, no_lines: int = 10, min_thickness: int = 1, max_thickness: int = 8):
        no_lines = int(no_lines)
        min_thickness = int(min_thickness)
        max_thickness = int(max_thickness)
        image_width = image.shape[1]
        vline_starts = np.random.randint(0, image_width, size=no_lines)
        vline_widths = np.random.randint(min_thickness, max_thickness, size=no_lines)
        for i in range(no_lines):
            start_idx = vline_starts[i]
            end_idx = min(vline_starts[i] + vline_widths[i], image_width)
            for channel in (0, 1, 2):
                image[start_idx:end_idx, :, channel] = np.random.randint(0, 255)
        return image

    @staticmethod
    def _apply_coloured_vlines(image, no_lines: int = 10, min_thickness: int = 1, max_thickness: int = 8,
                               colour_channel: int = 0):
        no_lines = int(no_lines)
        min_thickness = int(min_thickness)
        max_thickness = int(max_thickness)
        image_width = image.shape[1]
        vline_starts = np.random.randint(0, image_width, size=no_lines)
        vline_widths = np.random.randint(min_thickness, max_thickness, size=no_lines)
        for i in range(no_lines):
            start_idx = vline_starts[i]
            end_idx = min(vline_starts[i] + vline_widths[i], image_width)
            image[:, start_idx:end_idx, colour_channel] = 255
        return image

    @staticmethod
    def _replace_random_pixels(image, value: int = 255, replace_ratio: float = .1):
        noisy_pixels = int(image.size * replace_ratio)
        indices_to_modify = [np.random.randint(0, i - 1, noisy_pixels) for i in image.shape]
        modified_image = image.copy()
        modified_image[indices_to_modify[0], indices_to_modify[1], :] = [value, value, value]
        return modified_image


def _round_to_nearest_odd_int(x: Union[int, float]):
    x = int(x)
    if x % 2 != 1:
        x += 1
    return x


def _round_to_nearest_even_int(x: Union[int, float]):
    x = int(x)
    if x % 2 != 0:
        x += 1
    return x
