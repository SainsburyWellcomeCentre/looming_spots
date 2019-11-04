from matplotlib import pyplot as plt

from PyQt5.QtCore import QObject, QVariant, pyqtSlot

# For type hinting
from analysis.cell import Cell
from analysis.block import Block
from analysis.event_detection.gui.image_providers import PyplotImageProvider
from PyQt5.QtWidgets import QApplication
from PyQt5.QtQml import QQmlContext
from PyQt5.QtQuick import QQuickWindow


class DetectionInterface(QObject):
    """
    Abstract interface
    This class is meant to be sub-classed by the other classes of the module
    PlayerInterface, TrackerIface (base themselves to ViewerIface, CalibrationIface, RecorderIface)
    It supplies the base methods and attributes to register an object with video in qml.
    It also possesses an instance of ParamsIface to
    """

    def __init__(
        self,
        app,
        context,
        parent,
        display_name,
        image_provider,
        image_provider_name,
    ):
        """

        :param QApplication app:
        :param QQmlContext context:
        :param QQuickWindow parent:
        :param str display_name:
        :param PyplotImageProvider image_provider:
        :param str image_provider_name:
        """
        QObject.__init__(self, parent)
        self.app = (
            app
        )  # necessary to avoid QPixmap bug: must construct a QGuiApplication before
        self.ctx = context
        self.win = parent
        self.display_name = display_name
        self.provider_name = image_provider_name
        self._blocks = None
        self.angle = None
        self.fig = None
        self.current_block_id = 0
        self.current_trial_id = 0
        self.current_cell_type = "Pyramid"

        self.image_provider = image_provider
        self._set_display()

    def set_blocks(self, blocks, angle):
        self._blocks = blocks
        self.angle = angle
        self.set_n_blocks()
        self.set_n_trials()
        self.reload_ui_params()
        self.fig = plt.figure()
        self.image_provider._fig = self.fig

    def set_n_blocks(self):
        cell_id_control = self.win.findChild(QObject, "cellIdControl")
        cell_id_control.setMax(len(self._blocks) - 1)

    def set_n_trials(self):
        trial_id_control = self.win.findChild(QObject, "trialIdControl")
        trial_id_control.setMax(len(self._blocks[0]) - 1)

    def _set_display(self):
        """
        Gets the display from the qml code
        """
        self.display = self.win.findChild(QObject, self.display_name)

    @property
    def current_block(self):
        """

        :return:
        :rtype: Block
        """
        if self._blocks:
            return self._blocks[self.current_block_id]
        else:
            return None

    @property
    def current_params(self):
        """

        :return:
        :rtype: track_analysis.event_detection.detection_params.DetectionParams
        """
        if self.current_block:
            params = self.current_block.detection_params[self.angle]
            return params
        else:
            return None

    def reload_ui_params(self):
        ctrls = self.win.findChild(QObject, "mainControls")
        ctrls.reload()

    # FILTERING
    @pyqtSlot(result=int)
    def get_high_pass_n_pnts(self):
        if self.current_params:
            n_pnts = self.current_params.n_pnts_high_pass_filter
            return n_pnts

    @pyqtSlot(int)
    def set_high_pass_n_pnts(self, n_pnts):
        if self.current_params:
            self.current_params.n_pnts_high_pass_filter = n_pnts

    @pyqtSlot(result=int)
    def get_median_kernel_n_pnts(self):
        if self.current_params:
            return self.current_params.median_kernel_size

    @pyqtSlot(int)
    def set_median_kernel_n_pnts(self, n_pnts):
        if self.current_params:
            self.current_params.median_kernel_size = n_pnts

    @pyqtSlot()
    def save_filtering_params(self):
        if self.current_params:
            self.current_params.save_filtering_params()

    # DETECTION
    @pyqtSlot(result=int)
    def get_baseline_n_pnts(self):
        if self.current_params:
            return self.current_params.n_pnts_bsl

    @pyqtSlot(int)
    def set_baseline_n_pnts(self, n_pnts):
        if self.current_params:
            if 0 <= n_pnts < 50:
                self.current_params.n_pnts_bsl = n_pnts

    @pyqtSlot(result=int)
    def get_peak_n_pnts(self):
        if self.current_params:
            return self.current_params.n_pnts_peak

    @pyqtSlot(int)
    def set_peak_n_pnts(self, n_pnts):
        if self.current_params:
            if 0 <= n_pnts < 50:
                self.current_params.n_pnts_peak = n_pnts

    @pyqtSlot(result=int)
    def get_rt_n_pnts(self):
        if self.current_params:
            return self.current_params.n_pnts_rise_t

    @pyqtSlot(int)
    def set_rt_n_pnts(self, n_pnts):
        if self.current_params:
            self.current_params.n_pnts_rise_t = n_pnts

    @pyqtSlot(result=int)
    def get_detection_n_pnts(self):
        if self.current_params:
            return self.current_params.n_pnts_for_peak_detection

    @pyqtSlot(int)
    def set_detection_n_pnts(self, n_pnts):
        if self.current_params:
            self.current_params.n_pnts_for_peak_detection = n_pnts

    @pyqtSlot(result=float)
    def get_threshold(self):
        if self.current_params:
            return self.current_params.threshold

    @pyqtSlot(float)
    def set_threshold(self, thrsh):
        self.current_params.threshold = thrsh

    @pyqtSlot()
    def save_event_template_params(self):
        if self.current_params:
            self.current_params.save_event_template_params()

    # SDS
    @pyqtSlot(result=float)
    def get_n_sds(self):
        if self.current_params:
            return self.current_params.n_sds

    @pyqtSlot(float)
    def set_n_sds(self, n_sds):
        if self.current_params:
            self.current_params.n_sds = n_sds

    @pyqtSlot()
    def save_sd_params(self):
        if self.current_params:
            self.current_params.save_sd_params()

    @pyqtSlot()
    def detect(self):
        self.current_block.analyse(self.angle)
        plt.clf()
        self.current_block.plotter.plot_single_trial_detection(
            self.angle, self.current_trial_id
        )
        self.refresh_display()

    @pyqtSlot()
    def detect_all(self):
        for block in self._blocks:
            block.analyse(self.angle)

    @pyqtSlot(bool)
    def remove_current_cell(self, remove):
        if self.current_params:
            self.current_block.cell.skip = remove

    @pyqtSlot(result=bool)
    def get_remove_current_cell(self):
        if self.current_params:
            return self.current_block.cell.skip

    @pyqtSlot(bool)
    def remove_trial(self, remove):
        if self.current_params:
            for block in self._blocks:
                block.trials[self.current_trial_id].keep = not remove

    @pyqtSlot(result=bool)
    def get_remove_trial(self):
        if self.current_params:
            return not self.current_block.trials[self.current_trial_id].keep

    @pyqtSlot(result=int)
    def get_cell_id(self):
        return self.current_block_id

    @pyqtSlot(int)
    def set_cell_id(self, cell_id):
        self.current_block_id = cell_id

    @pyqtSlot(result=int)
    def get_trial_id(self):
        return self.current_trial_id

    @pyqtSlot(int)
    def set_trial_id(self, trial_id):
        self.current_trial_id = trial_id

    @pyqtSlot(str)
    def set_cell_type(self, cell_type):
        self.current_cell_type = cell_type
        if self.current_params:
            self.current_block.cell.cell_type = self.current_cell_type

    @pyqtSlot(result=int)
    def get_cell_type_index(self):
        if self.current_params:
            return Cell.CELL_TYPES.index(self.current_block.cell.cell_type)

    def refresh_display(self):
        self.display.reload()

    @pyqtSlot(result=int)
    def get_n_events(self):
        if self.current_params:
            return len(self.current_block.trials[self.current_trial_id].events)

    @pyqtSlot(result=str)
    def get_average_amplitude(self):
        if self.current_params:
            ampl = self.current_block.trials[
                self.current_trial_id
            ].events.average_amplitude()
            return "{0:.2f}".format(ampl)

    @pyqtSlot(result=str)
    def get_integral(self):
        if self.current_params:
            integral = self.current_block.get_events_integral(
                self.angle, self.current_trial_id
            )
            return "{0:.2f}".format(integral)

    @pyqtSlot()
    def init_ui(self):
        cell_id_ctrl = self.win.findChild(QObject, "cellIdControl")
        cell_id_ctrl.reload()
