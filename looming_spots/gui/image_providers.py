from PyQt5.QtCore import QSize
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtQuick import QQuickImageProvider

import numpy as np


class PyplotImageProvider(QQuickImageProvider):
    """
    This class implements TrackingImageProvider for pyplot graphs.
    If supplied it will use the pyplot graph to get the next image.
    If it cannot get an image, a random image (noise) of the proper size will be generated as
    a place holder.
    """

    def __init__(self, requestedImType="img", fig=None):
        """
        :param string requestedImType: The type of image to return to the QT interface (one of ['img', 'pixmap'])
        """

        if requestedImType == "img":
            imType = QQuickImageProvider.Image
        elif requestedImType == "pixmap":
            imType = QQuickImageProvider.Pixmap
        else:
            raise NotImplementedError(
                "Unknown type: {}".format(requestedImType)
            )
        QQuickImageProvider.__init__(self, imType)

        self._fig = fig

    def requestPixmap(self, id, qSize):
        """
        Returns the next image formated as a pixmap for QT with the associated QSize object
        """
        qimg, qSize = self.requestImage(id, qSize)
        img = QPixmap.fromImage(qimg)

        return img, qSize

    def getArray(self):
        """
        Return the graph drawn on self._fig as a raw rgb image (numpy array)
        """
        self._fig.canvas.draw()
        width, height = self._fig.canvas.get_width_height()
        img = self._fig.canvas.tostring_rgb()
        img = np.fromstring(img, dtype=np.uint8).reshape(height, width, 3)
        return img

    def getBaseImg(self, size):
        """
        The method common to requestPixmap() and requestImage() to get the image from the graph before formatting

        :param tuple size: The desired image size
        :returns: the output image
        :rtype: QImage
        """
        if self._fig is not None:
            img = self.getArray()
            size = img.shape[:2]
        else:
            img = self.getRndmImg(size)
        w, h = size
        qimg = QImage(img, h, w, QImage.Format_RGB888)

        return qimg

    def getRndmImg(self, size):
        """
        Generates a random image of size

        :param tuple size: The desired output image size
        :returns: The image
        """
        img = np.random.random(size)
        img *= 125
        img = img.astype(np.uint8)
        img = np.dstack([img] * 3)

        return img

    def requestImage(self, id, qSize):
        """
        Returns the next image formated as a QImage for QT with the associated QSize object
        """
        size = self.getSize(qSize)
        qimg = self.getBaseImg(size)
        return qimg, QSize(*size)

    def getSize(self, qSize):
        """
        Gets the qSize as a tuple of (width, height) (for openCV which flips x and y dimensions)
        If the input size is invalid it defaults to (512, 512)

        :param QSize qSize: The QT size object to convert
        :returns: size
        :rtype: tuple
        """

        return (
            (qSize.width(), qSize.height()) if qSize.isValid() else (512, 512)
        )
