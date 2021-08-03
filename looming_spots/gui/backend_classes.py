import pandas as pd
from PyQt5.QtCore import QObject, pyqtSlot
from looming_spots.db import experimental_log
from looming_spots.db.experimental_log import (
    get_comparator_functions,
    filter_df,
)


class Logger(QObject):
    """
    A qml object for logging the events
    In production, the standard out is redirected to it.
    It is not meant to be interacted with directly (use print() instead)
    """

    def __init__(self, context, parent=None, log_object_name="log"):
        QObject.__init__(self, parent)
        self.win = parent
        self.ctx = context
        self.log = self.win.findChild(QObject, log_object_name)

    def write(self, text):
        """
        The method to make it compatible with sys.stdout
        The text gets printed in the corresponding qml component

        :param string text: The text to append at the end of the current qml component text
        """
        if text:
            previous_text = self.log.property("text")
            output_text = "{}\n>>>{}".format(previous_text, text)

            self.log.setProperty("text", output_text)


class LoomTrialGroupBackend(QObject):
    """
    The QObject derived class that stores most of the parameters from the graphical interface
    for the other QT interfaces
    """

    def __init__(self, app, context, parent):
        """
        :param app: The QT application
        :param context: context defines what can know about and access the backend class
        :param parent: the parent window
        """
        QObject.__init__(self, parent)
        self.app = app  # necessary to avoid QPixmap bug: Must construct a QGuiApplication before
        self.win = parent
        self.ctx = context

        self.data = experimental_log.load_df()

        self._set_defaults()
        self.condition_dictionaries = [{}, {}]
        self.dfs = [pd.DataFrame(), pd.DataFrame()]
        self.current_key = None

        self.test_phases = []
        self.exclude_test_phases = ["pre_test", "habituation", "post_test"]
        self.n_records = 0

    def _set_defaults(self):
        """
        Reset the parameters to default.
        To customise the defaults, users should do this in the config file.
        """
        pass

    @pyqtSlot(int, str, str, str)
    def update_condition_dictionary(self, idx, key, value, comparator):
        comparator_value_string = "{} {}".format(comparator, value)
        print(key, comparator_value_string)
        self.condition_dictionaries[idx].setdefault(
            key, comparator_value_string
        )

    @pyqtSlot()
    def reset_conditions(self):
        self.condition_dictionaries = [{}, {}]
        self.dfs = [pd.DataFrame(), pd.DataFrame()]

    def keys(self):
        """
        :return list keys: all possible options from database for selection menu
        """
        keys = list(self.data.keys())
        return keys

    @pyqtSlot(int, result=str)
    def get_key_at(self, idx):
        """

        :param idx:
        :return: the currently selected key from the database
        """
        return self.keys()[idx]

    @pyqtSlot(result=int)
    def get_n_keys(self):
        """
        :return int: the total number of keys
        """
        return len(self.keys())

    @pyqtSlot(str, result=str)
    def set_current_key(self, current_key):
        self.current_key = current_key

    @pyqtSlot(str, result=str)
    def get_comparators(self):
        return list(get_comparator_functions().keys())

    @pyqtSlot(int, result=str)
    def get_comparator_at(self, idx):
        return self.get_comparators()[idx]

    @pyqtSlot(result=int)
    def get_n_comparators(self):
        return len(self.get_comparators())

    @pyqtSlot(str, result=str)
    def get_options(self):
        options = list(self.data[self.current_key].unique())
        options = [str(x) for x in options]
        return options

    @pyqtSlot(int, result=str)
    def get_option_at(self, idx):
        return self.get_options()[idx]

    @pyqtSlot(result=int)
    def get_n_options(self):
        return len(self.get_options())

    @pyqtSlot(str)
    def test_type_present(self, test_type):
        if test_type not in self.test_phases:
            self.test_phases.append(test_type)
            self.exclude_test_phases.remove(test_type)
        else:
            self.exclude_test_phases.append(test_type)
            self.test_phases.remove(test_type)

    @pyqtSlot()
    def reset_db(self):
        self.data = experimental_log.load_df()

    @pyqtSlot()
    def filter_data(self):
        self.reset_db()
        relevant_mouse_ids = (
            experimental_log.get_mouse_ids_with_test_combination(
                self.data, self.test_phases, self.exclude_test_phases
            )
        )
        self.data = self.data[self.data.mouse_id.isin(relevant_mouse_ids)]

    @pyqtSlot(result=int)
    def n_records_displayed(self):
        print("number of records on display: {}".format(self.n_records))
        return self.n_records

    @pyqtSlot(int, result=str)
    def display_table(self, idx):
        self.filter_data()
        print(self.condition_dictionaries[0])
        self.dfs[0] = filter_df(self.data, self.condition_dictionaries[0])
        self.n_records = len(self.dfs[0])
        print(
            "{} records matching these criteria were found".format(
                self.n_records
            )
        )
        self.dfs[1] = filter_df(self.data, self.condition_dictionaries[1])
        return self.dfs[idx].to_html()
