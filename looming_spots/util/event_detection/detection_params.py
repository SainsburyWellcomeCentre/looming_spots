import os

import pandas as pd
from configobj import ConfigObj


class DetectionParams(object):
    DEFAULT_CONFIG_PATH = os.path.normpath(
        os.path.expanduser("~/.sepi_analysis.conf")
    )
    DEFAULT_CONF = ConfigObj(
        DEFAULT_CONFIG_PATH,
        encoding="UTF8",
        indent_type=" " * 4,
        unrepr=True,
        create_empty=True,
        write_empty_values=True,
    )
    DEFAULT_CONF.reload()

    def __init__(self, config_file_path=None):
        """

        :param str config_file_path:
        """
        if config_file_path is None:
            self.config_path = DetectionParams.DEFAULT_CONFIG_PATH
            self.conf = DetectionParams.DEFAULT_CONF
        elif config_file_path.endswith("conf"):
            self.from_conf(config_file_path)
        elif config_file_path.endswith("csv"):
            self.from_csv(config_file_path)

        self.threshold = self.conf["event_template"]["threshold"]
        self.n_pnts_bsl = self.conf["event_template"]["n_pnts_bsl"]
        self.n_pnts_peak = self.conf["event_template"]["n_pnts_peak"]
        self.n_pnts_rise_t = self.conf["event_template"]["n_pnts_rise_t"]
        self.n_pnts_for_peak_detection = self.conf["event_template"][
            "n_pnts_for_peak_detection"
        ]

        self.n_sds = self.conf["sds"]["n_sds"]

        self.n_pnts_high_pass_filter = self.conf["filtering"][
            "n_pnts_high_pass_filter"
        ]
        self.median_kernel_size = self.conf["filtering"]["median_kernel_size"]

    def __str__(self):
        return str(self.to_df())

    def from_conf(self, config_file_path):
        self.config_path = os.path.normpath(
            os.path.expanduser(config_file_path)
        )
        cfg = ConfigObj(
            self.config_path,
            encoding="UTF8",
            indent_type=" " * 4,
            unrepr=True,
            create_empty=True,
            write_empty_values=True,
        )
        cfg.reload()
        self.conf = cfg

    def from_csv(self, config_file_path):
        df = pd.read_csv(config_file_path)
        self.config_path = os.path.normpath(
            os.path.expanduser(config_file_path)
        )
        self.config_path = os.path.splitext(self.config_path)[0] + ".conf"
        self.conf = ConfigObj(
            self.config_path,
            encoding="UTF8",
            indent_type=" " * 4,
            unrepr=True,
            create_empty=True,
            write_empty_values=True,
        )  # TEST:
        self.conf["event_template"] = {}
        self.conf["event_template"]["threshold"] = df["threshold"][0]
        self.conf["event_template"]["n_pnts_bsl"] = df["n_pnts_bsl"][0]
        self.conf["event_template"]["n_pnts_peak"] = df["n_pnts_peak"][0]
        self.conf["event_template"]["n_pnts_rise_t"] = df["n_pnts_rise_t"][0]
        self.conf["event_template"]["n_pnts_for_peak_detection"] = df[
            "n_pnts_for_peak_detection"
        ][0]
        self.conf["sds"] = {}
        self.conf["sds"]["n_sds"] = df["n_sds"][0]
        self.conf["filtering"] = {}
        self.conf["filtering"]["n_pnts_high_pass_filter"] = df[
            "n_pnts_high_pass_filter"
        ][0]
        self.conf["filtering"]["median_kernel_size"] = df[
            "median_kernel_size"
        ][0]

    def save_conf(self):
        self.conf.write()

    def save_event_template_params(self):
        evt_template = self.conf["event_template"]
        evt_template["threshold"] = self.threshold
        evt_template["n_pnts_bsl"] = self.n_pnts_bsl
        evt_template["n_pnts_peak"] = self.n_pnts_peak
        evt_template["n_pnts_rise_t"] = self.n_pnts_rise_t
        evt_template[
            "n_pnts_for_peak_detection"
        ] = self.n_pnts_for_peak_detection

        self.save_conf()

    def save_filtering_params(self):
        filtering_params = self.conf["filtering"]
        filtering_params[
            "n_pnts_high_pass_filter"
        ] = self.n_pnts_high_pass_filter
        filtering_params["median_kernel_size"] = self.median_kernel_size

        self.save_conf()

    def save_sd_params(self):
        self.conf["sds"]["n_sds"] = self.n_sds

        self.save_conf()

    def to_df(self):
        return pd.DataFrame(
            {
                "n_pnts_high_pass_filter": self.n_pnts_high_pass_filter,
                "median_kernel_size": self.median_kernel_size,
                "threshold": self.threshold,
                "n_pnts_bsl": self.n_pnts_bsl,
                "n_pnts_peak": self.n_pnts_peak,
                "n_pnts_rise_t": self.n_pnts_rise_t,
                "n_pnts_for_peak_detection": self.n_pnts_for_peak_detection,
                "n_sds": self.n_sds,
            },
            columns=(
                "n_pnts_high_pass_filter",
                "median_kernel_size",
                "threshold",
                "n_pnts_bsl",
                "n_pnts_peak",
                "n_pnts_rise_t",
                "n_pnts_for_peak_detection",
                "n_sds",
            ),
            index=[0],
        )
