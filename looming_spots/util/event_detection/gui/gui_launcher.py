# -*- coding: utf-8 -*-
"""
**************
The GUI module
**************

Creates the graphical interface


.. note::
    This module depends on importing OpenGL.GL although it doesn't uses it directly but it
    is used by the Qt interface.

:author: crousse
"""
import os
import sys

# WARNING: Hack necessary to get qtQuick working
from OpenGL import GL

from PyQt5.QtQml import QQmlApplicationEngine
from PyQt5.QtWidgets import QApplication

from looming_spots.analyse.event_detection.gui.image_providers import (
    PyplotImageProvider,
)
from looming_spots.analyse.event_detection.gui.detection_interface import (
    DetectionInterface,
)


class GuiError(Exception):
    pass


def get_main_window(app_engine):
    if sys.platform == "win32":
        base_dir = os.path.abspath(
            os.path.normpath("./analyse/event_detection/gui/")
        )  # FIXME: should be conf based
    else:
        base_dir = os.path.abspath(
            os.path.normpath("./event_detection/gui/")
        )  # FIXME: should be conf based
    qml_source_path = os.path.join(base_dir, "qml", "Detection", "main.qml")
    if not os.path.isfile(qml_source_path):
        raise GuiError(
            "Qml code not found at {}, please verify your installation".format(
                qml_source_path
            )
        )
    app_engine.load(qml_source_path)
    try:
        win = app_engine.rootObjects()[0]
    except IndexError:
        raise GuiError("Could not start the QT GUI")
    return win


def main(blocks=None, angle=None):
    app = QApplication(sys.argv)
    app_engine = QQmlApplicationEngine()
    context = app_engine.rootContext()

    # WARNING: ALL THE ADDIMAGEPROVIDER LINES BELOW ARE REQUIRED TO MAKE
    # WARNING: QML BELIEVE THE PROVIDER IS VALID BEFORE ITS CREATION
    analysis_image_provider = PyplotImageProvider(fig=None)
    app_engine.addImageProvider("analysisprovider", analysis_image_provider)

    win = get_main_window(app_engine)

    detector = DetectionInterface(
        app,
        context,
        win,
        display_name="detectionGraph",
        image_provider=analysis_image_provider,
        image_provider_name="analysisprovider",
    )
    context.setContextProperty("py_detector", detector)

    if blocks is not None:
        detector.set_blocks(blocks, angle)

    win.show()
    # detector.detect()  # to display cell 0
    detector.init_ui()
    app.exec_()


if __name__ == "__main__":
    main()
