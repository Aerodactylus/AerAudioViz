import cv2
from matplotlib import pyplot as plt


class BaseImage:

    def __init__(self, image_path: str, size=(1080, 1080)):
        """

        :param image_path:
        """
        self.image_path = image_path
        self.bgr_image = cv2.resize(cv2.imread(self.image_path), size)
        self.rgb_image = cv2.cvtColor(self.bgr_image, cv2.COLOR_BGR2RGB)
        self.hsv_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2HSV)

    def show(self):
        """

        :return: matplotlib.image.AxesImage object
        """
        return plt.imshow(self.rgb_image)

    def show_grayscale(self):
        return plt.imshow(self.hsv_image[:, :, 2], cmap='gray')

    def show_hue(self):
        plot = plt.imshow(self.hsv_image[:, :, 0], cmap='gray')
        plt.colorbar()
        return plot

    def show_saturation(self):
        plot = plt.imshow(self.hsv_image[:, :, 1], cmap='gray')
        plt.colorbar()
        return plot

    def show_red_channel(self):
        plot = plt.imshow(self.rgb_image[:, :, 0], cmap='gray')
        plt.colorbar()
        return plot

    def show_green_channel(self):
        plot = plt.imshow(self.rgb_image[:, :, 1], cmap='gray')
        plt.colorbar()
        return plot

    def show_blue_channel(self):
        plot = plt.imshow(self.rgb_image[:, :, 2], cmap='gray')
        plt.colorbar()
        return plot
