import os
import sys

if sys.platform.startswith("linux"):
    from OpenGL import GL

from PyQt5.QtQml import QQmlApplicationEngine
from PyQt5.QtWidgets import QApplication

from looming_spots.gui.backend_classes import LoomTrialGroupBackend, Logger

DEBUG = False


if __name__ == "__main__":

    # QT INITIALISATION STEPS
    app = QApplication(sys.argv)
    appEngine = QQmlApplicationEngine()
    context = appEngine.rootContext()

    # POINT TO QML FILEs
    conf = {"shared_directory": "./"}

    qml_source_path = os.path.join(
        conf["shared_directory"], "qml", "gui_qtquick", "gui_qtquick.qml"
    )

    if not os.path.isfile(qml_source_path):
        raise ValueError(
            "Qml code not found at {}, please verify your installation".format(
                qml_source_path
            )
        )
    appEngine.load(qml_source_path)

    # CREATE WINDOW
    try:
        win = appEngine.rootObjects()[0]
    except IndexError:
        raise ValueError("Could not start the QT GUI")

    # CREATE AND REGISTER PYTHON BACKENDS

    if not DEBUG:
        logger = Logger(context, win, "log")
        sys.stdout = logger

    print("Hello world")

    backend = LoomTrialGroupBackend(app, context, win)
    context.setContextProperty("py_iface", backend)

    win.show()

    sys.exit(app.exec_())
