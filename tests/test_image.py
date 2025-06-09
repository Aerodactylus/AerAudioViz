from matplotlib.image import AxesImage
import numpy as np
import pytest

from aeraudioviz.image import BaseImage, ImageModifiers


class TestBaseImage:

    IMAGE_FILE = 'tests/data/aero_head_lq.png'
    NATIVE_SIZE = (150, 150)
    LARGE_SIZE = (500, 500)
    
    @classmethod
    def setup_class(cls):
        cls.img = BaseImage(cls.IMAGE_FILE, size=cls.NATIVE_SIZE)
        cls.img_resized = BaseImage(cls.IMAGE_FILE, size=cls.LARGE_SIZE)

    def test_load_base_image(self):
        assert isinstance(self.img.bgr_image, np.ndarray)
        assert isinstance(self.img.rgb_image, np.ndarray)
        assert isinstance(self.img.hsv_image, np.ndarray)
        assert self.img.bgr_image.shape == (self.NATIVE_SIZE[0], self.NATIVE_SIZE[1], 3)
    
    def test_resizing(self):
        assert self.img_resized.bgr_image.shape == (self.LARGE_SIZE[0], self.LARGE_SIZE[1], 3)

    def test_image_show_methods(self):
        assert isinstance(self.img.show(), AxesImage)
        assert isinstance(self.img.show_blue_channel(), AxesImage)
        assert isinstance(self.img.show_grayscale(), AxesImage)
        assert isinstance(self.img.show_green_channel(), AxesImage)
        assert isinstance(self.img.show_hue(), AxesImage)
        assert isinstance(self.img.show_red_channel(), AxesImage)
        assert isinstance(self.img.show_saturation(), AxesImage)
        assert isinstance(self.img_resized.show(), AxesImage)
