import QtQuick 2.3
import QtQuick.Window 2.2
import QtQuick.Controls 1.4
import QtQuick.Layouts 1.2

import "basic_types"
import "style"

Window {
    visible: true
    width: 900
    height: 650

    Rectangle {
        anchors.fill: parent
        color: Theme.background
        Row {
            anchors.fill: parent
            anchors.margins: 15
            spacing: 25
            Column {
                id: main_controls
                objectName: 'mainControls'
                anchors.top: parent.top
                anchors.topMargin: 15
                anchors.bottom: parent.bottom
                width: 200
                Layout.maximumWidth: 300

                function reload() {
                    filtering_controls.reload();
                    event_template_controls.reload();
                    sd_detection_controls.reload();
                }

                Column {
                    id: filtering_controls
                    width: parent.width
                    height: 100
                    function reload() {
                        highPassCtrl.reload();
                        medianKernelCtrl.reload();
                    }

                    Label {
                        anchors.horizontalCenter: parent.horizontalCenter
                        width: parent.width
                        text: "Filtering:"
                        color: "#ffffff"
                        horizontalAlignment: Text.AlignHCenter
                    }
                    IntInput {
                        id: highPassCtrl
                        label: "High pass"
                        width: parent.width
                        boxWidth: 65
                        tooltip: "Set the number of points for the high pass filter"
                        value: py_detector.get_high_pass_n_pnts();
                        decimals: 0
                        onValueChanged: py_detector.set_high_pass_n_pnts(value);
                        function reload() {
                            value = py_detector.get_high_pass_n_pnts();
                        }
                    }
                    IntInput {
                        id: medianKernelCtrl
                        label: "Median kernel"
                        width: parent.width
                        boxWidth: 65
                        tooltip: "Set the number of points for the median filtering kernel size"
                        value: py_detector.get_median_kernel_n_pnts();
                        decimals: 0
                        onValueChanged: py_detector.set_median_kernel_n_pnts(value);
                        function reload() {
                            value = py_detector.get_median_kernel_n_pnts();
                        }
                    }
                    Button {
                        id: filterSaveBtn
                        width: parent.width
                        text: "Save as default"
                        tooltip: "Save these parameters for reuse the next time the  software is started"
                        onClicked: py_detector.save_filtering_params();
                    }
                }
                Column {
                    id: event_template_controls
                    width: parent.width
                    height: 170
                    function reload() {
                        bslNPntsCtrl.reload();
                        peakNPntsCtrl.reload();
                        rtNPntsCtrl.reload();
                        peakDetectionNPntsCtrl.reload();
                        thresholdCtrl.reload();
                    }

                    Label {
                        anchors.horizontalCenter: parent.horizontalCenter
                        width: parent.width
                        text: "Event template:"
                        color: "#ffffff"
                        horizontalAlignment: Text.AlignHCenter
                    }
                    IntInput {
                        id: bslNPntsCtrl
                        label: "Baseline"
                        width: parent.width
                        boxWidth: 65
                        tooltip: "Set the number of points for the baseline"
                        value: py_detector.get_baseline_n_pnts();
                        decimals: 0
                        onValueChanged: py_detector.set_baseline_n_pnts(value);
                        function reload() {
                            value = py_detector.get_baseline_n_pnts();
                        }
                    }
                    IntInput {
                        id: peakNPntsCtrl
                        label: "Peak"
                        width: parent.width
                        boxWidth: 65
                        tooltip: "Set the number of points for the peak"
                        value: py_detector.get_peak_n_pnts();
                        decimals: 0
                        onValueChanged: py_detector.set_peak_n_pnts(value);
                        function reload() {
                            value = py_detector.get_peak_n_pnts();
                        }
                    }
                    IntInput {
                        id: rtNPntsCtrl
                        label: "Rise time"
                        width: parent.width
                        boxWidth: 65
                        tooltip: "Set the number of points for the rise time"
                        value: py_detector.get_rt_n_pnts();
                        decimals: 0
                        onValueChanged: py_detector.set_rt_n_pnts(value);
                        function reload() {
                            value = py_detector.get_rt_n_pnts();
                        }
                    }
                    IntInput {
                        id: peakDetectionNPntsCtrl
                        label: "Peak detection"
                        width: parent.width
                        boxWidth: 65
                        tooltip: "Set the number of points for the detection of events after the start"
                        value: py_detector.get_detection_n_pnts();
                        decimals: 0
                        onValueChanged: py_detector.set_detection_n_pnts(value);
                        function reload() {
                            value = py_detector.get_detection_n_pnts();
                        }
                    }
                    IntInput {
                        id: thresholdCtrl
                        label: "Threshold"
                        width: parent.width
                        boxWidth: 65
                        tooltip: "Set the Threshold"
                        value: py_detector.get_threshold();
                        decimals: 3
                        stepSize: 0.001
                        onValueChanged: py_detector.set_threshold(value);
                        function reload() {
                            value = py_detector.get_threshold();
                        }
                    }

                    Button {
                        id: detectionParamsSaveBtn
                        width: parent.width
                        text: "Save as default"
                        tooltip: "Save these parameters for reuse the next time the  software is started"
                        onClicked: py_detector.save_event_template_params();
                    }
                }
                Column {
                    id: sd_detection_controls
                    width: parent.width
                    height: 100
                    function reload() {
                        nSdsCtrl.reload();
                    }

                    Label {
                        anchors.horizontalCenter: parent.horizontalCenter
                        width: parent.width
                        text: "SD parameters:"
                        color: "#ffffff"
                        horizontalAlignment: Text.AlignHCenter
                    }
                    IntInput {
                        id: nSdsCtrl
                        label: "N SDS"
                        width: parent.width
                        boxWidth: 65
                        tooltip: "Set the number of standard deviations for accepting events"
                        value: py_detector.get_n_sds();
                        decimals: 1
                        stepSize: 0.1
                        onValueChanged: py_detector.set_n_sds(value);
                        function reload() {
                            value = py_detector.get_n_sds();
                        }
                    }
                    Button {
                        id: sdParamsSaveBtn
                        width: parent.width
                        text: "Save as default"
                        tooltip: "Save these parameters for reuse the next time the  software is started"
                        onClicked: py_detector.save_sd_params();
                    }
                }
                Button {
                    width: parent.width
                    height: 25
                    text: "DETECT"
                    tooltip: "Start event detection"
                    onClicked: {
                        py_detector.detect();
                        feedbackCtrls.reload();
                    }
                }
                Button {
                    width: parent.width
                    height: 25
                    text: "DETECT ALL"
                    tooltip: "Start event detection"
                    onClicked: {
                        py_detector.detect_all();
                        feedbackCtrls.reload();
                    }
                }
            }
            Column {
                width: 600
                spacing: 15
                Row {
                    anchors.left: parent.left
                    anchors.right: parent.right
                    height: 80
                    Grid {
                        columns: 2
                        rows: 2
                        anchors.top: parent.top
                        anchors.bottom: parent.bottom
                        width: 450
                        spacing: 15
                        BoolInput {
                            id: removeCellCtrl
                            width: parent.width / 2 - parent.spacing
                            height: 25
                            boxWidth: 60
                            tooltip: "Remove the current cell from the analysis"
                            label: 'Remove cell'
                            checked: py_detector.get_remove_current_cell();
                            onCheckedChanged: {
                                py_detector.remove_current_cell(checked);
                            }
                            function reload() {
                                checked = py_detector.get_remove_current_cell();
                            }
                        }
                        BoolInput {
                            id: removeTrialCtrl
                            width: parent.width / 2 - parent.spacing
                            height: 25
                            tooltip: "Remove the current trial from the analysis"
                            label: 'Remove trial'
                            checked: py_detector.get_remove_trial();
                            onCheckedChanged:  {
                                py_detector.remove_trial(checked)
                            }
                            function reload() {
                                checked = py_detector.get_remove_trial();
                            }
                        }
                        IntInput {
                            id: cellIdCtrl
                            objectName: 'cellIdControl'
                            label: 'Cell'
                            width: parent.width / 2 - parent.spacing
                            boxWidth: 65
                            tooltip: "Set the current cell"
                            value: py_detector.get_cell_id();
                            decimals: 0
                            onValueChanged: {
                                reload();
                            }
                            function setMax(_max) {
                                maximumValue = _max;
                            }
                            function reload() {
                                py_detector.set_cell_id(value);
                                cellTypeSelector.reload();
                                removeCellCtrl.reload();
                                main_controls.reload();
                                py_detector.detect();
                                feedbackCtrls.reload();
                            }
                        }
                        IntInput {
                            objectName: 'trialIdControl'
                            label: 'Trial'
                            width: parent.width / 2 - parent.spacing
                            boxWidth: 65
                            tooltip: "Set the current trial"
                            value: py_detector.get_trial_id();
                            decimals: 0
                            onValueChanged: {
                                py_detector.set_trial_id(value);
                                py_detector.detect();
                                removeTrialCtrl.reload();
                            }
                            function setMax(_max) {
                                maximumValue = _max;
                            }
                        }

                    }
                    ComboBox {
                        id: cellTypeSelector
                        anchors.verticalCenter: parent.verticalCenter
                        height: 25
                        width: 100
                        editable: true
                        model: ["Pyramid", "Interneuron"]
                        onCurrentTextChanged: { py_detector.set_cell_type(currentText); }
                        currentIndex: py_detector.get_cell_type_index();
                        onPressedChanged: {
                            if (pressed) {
                                py_detector.set_cell_type(currentText);
                            }
                        }
                        function reload() {
                            currentIndex = py_detector.get_cell_type_index();
                        }
                    }

                }
                Image {
                    id: detectionGraph
                    objectName: 'detectionGraph'

                    width: 600
                    height: 400

                    source: "image://analysisprovider/img";
                    function reload() {
                        var oldSource = source;
                        source = "";
                        source = oldSource;
                    }
                    sourceSize.height: height
                    sourceSize.width: width
                    cache: false
                }

                Column {
                    id: feedbackCtrls
                    anchors.left: parent.left
                    anchors.right: parent.right
                    spacing: 5
                    function reload() {
                        nEventsCtrl.reload();
                        avgAmplCtrl.reload();
                        integralCtrl.reload();
                    }

                    IntLabel {
                        id: nEventsCtrl
                        tooltip: "The number of events detected above"
                        label: 'Number of events'

                        value: py_detector.get_n_events();
                        function reload() {
                            value = py_detector.get_n_events();
                        }
                    }
                    IntLabel {
                        id: avgAmplCtrl
                        tooltip: "Average amplitude of the detected events"
                        label: "Peak amplitude"

                        value: py_detector.get_average_amplitude();
                        function reload() {
                            value = py_detector.get_average_amplitude();
                        }
                    }
//                    IntLabel {
//                        tooltip: "Average smoothness of the evens decay"
//                        label: "Smoothness"

//                        value: py_detector.get_n_events();
//                        function reload() {
//                            value = py_detector.get_n_events();
//                        }
//                    }
                    IntLabel {
                        id: integralCtrl
                        tooltip: "Average area under the curve of the events"
                        label: "Integral"

                        value: py_detector.get_integral();
                        function reload() {
                            value = py_detector.get_integral();
                        }
                    }
                }

            }
        }
    }
}

