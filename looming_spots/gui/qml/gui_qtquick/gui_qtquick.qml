import QtQuick 2.5
import QtQuick.Window 2.2
import QtQuick.Controls 1.3


Window {
    id: window1
    visible: true

    width: 1200
    height: 1200

    MouseArea {
        anchors.rightMargin: 0
        anchors.bottomMargin: 0
        anchors.leftMargin: 0
        anchors.topMargin: 0
        anchors.fill: parent
        onClicked: {}

        Row {
            anchors.left: parent.left
            anchors.leftMargin: 20
            anchors.bottom: queryElements.bottom

            CheckBox {
                id: checkBox1
                text: qsTr("pre_test")
                anchors.left: queryElements.left
                onClicked: {
                    py_iface.test_type_present(checkBox1.text);
                    }
            }


            CheckBox {
                id: checkBox2
                anchors.left: checkBox1.right
                anchors.leftMargin: 20
                text: qsTr("habituation")
                onClicked: {py_iface.test_type_present(checkBox2.text) }

            }

            CheckBox {
                id: checkBox3
                anchors.left: checkBox2.right
                anchors.leftMargin: 20
                text: qsTr("post_test")
                onClicked: {py_iface.test_type_present(checkBox3.text) }
            }

        }

        Button {
            id: button1
            x: 663
            y: 119
            width: 486
            height: 214
            text: qsTr("Button")
            onClicked: {

                console.log('Ohhhhhh click click click')
                button1.set_text('Ohhhhhh sweet baby jesus click click click')
                }

            function set_text(txt){
                button1.text = txt
            }
        }
    }

    Text {
        id: txt1
        color: "#000000"

        text: "LOOM QUERY"
        textFormat: Text.AutoText
        style: Text.Raised
        font.family: "Helvetica"
        font.bold: true
        styleColor: "#797979"

        anchors.top: parent.top
        anchors.right: parent.right
        anchors.rightMargin: 10

        anchors.topMargin: 0


        font.pointSize: 32


        onTextChanged: {
            console.log("I've changed !")
        }
    }

    Row {

        id: updateReset

        width: 150
        height: 50
        anchors.left: setConditions.left
        anchors.bottom: setConditions.top

        ToolButton {
                    id: updateSettings
                    objectName: "updateSettings"

                    width: updateReset.width
                    height: updateReset.height
                    anchors.top: parent.top
                    anchors.bottomMargin: 10

                    text: "Load Options"
                    rotation: 0
                    activeFocusOnPress: true
                    tooltip: "Update the list in the combo boxes"

                    onClicked: {
                        console.log("Ooh I've been clicked");

                        optionGrid1.reload();
                        optionGrid2.reload();
                        optionGrid3.reload();
                    }
            }


    }

    Column {
            id: setConditions

            width: 150
            height: 50
            anchors.top: queryElements.top
            anchors.left: queryElements.right
            anchors.leftMargin: 10


        ToolButton {
                    id: updateCondition1

                    width: setConditions.width
                    height: setConditions.height
                    anchors.top: addCondition.bottom
                    anchors.topMargin: 10
                    text: "Set Condition"
                    rotation: 0
                    tooltip: "Sets conditions for first group"
                    onClicked: queryElements.get_keys(0)
                    }


        ToolButton {
                    id: reset

                    width: updateCondition1.width
                    height: updateCondition1.height

                    anchors.horizontalCenter: txt1.horizontalCenter
                    anchors.top: updateCondition1.bottom
                    anchors.topMargin: 10

                    text: "Reset Conditions"
                    tooltip: "clears dictionaries"
                    onClicked: {
                        py_iface.reset_conditions()
                        py_iface.reset_db()
                    }
                    }

        ToolButton {
                    id: addCondition

                    width: updateCondition1.width
                    height: updateCondition1.height

                    anchors.top: queryElements.top
                    anchors.left: queryElements.right

                    anchors.topMargin: 10
                    anchors.leftMargin: 10

                    text: "Add Condition"
                    tooltip: "adds new combobox row"
                    onClicked: {

                                var component = Qt.createComponent("OptionGrid2.qml");
                                if (component.status == Component.Ready)
                                    component.createObject(queryElements, {"x":0, "y": 100});
                                }

                     }

}

    ToolButton {
        id: compute

        width: updateCondition1.width
        height: updateCondition1.height

        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 20
        anchors.rightMargin: 50
        anchors.top: displayStats.bottom

        text: "compute"
        tooltip: "computes"
        onClicked: displayStats.set_text(py_iface.compare())
    }


    ToolButton {
        id: display

        width: updateCondition1.width
        height: updateCondition1.height

        anchors.right: compute.left
        anchors.top: displayStats.bottom
        anchors.bottom: parent.bottom

        anchors.bottomMargin: 20

        text: "display trials"
        tooltip: "display trials"
        onClicked: {
            resultsTable1.set_text(py_iface.display_table(0));
            resultsTable2.set_text(py_iface.display_table(1));
            nRecords.set_text(py_iface.n_records_displayed())
            py_iface.generate_plots();
            analysisImage1.reload();
            analysisImage2.reload();

        }
    }



    Column{
        id: queryElements
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.topMargin: 130
        anchors.leftMargin: 20
        width: parent.width/3
        height: 184
        rotation: 0
        spacing: 2

        function get_keys(dict_idx) {
            for (var i = 0; i < children.length; i++)
            {
                py_iface.update_condition_dictionary(dict_idx, children[i].get_key(), children[i].get_value(), children[i].get_comparator());

            }
        }
        function reset(){
            for (var i = 0; i < children.length; i++)
            {
                children[i].reload()
            }
        }

        OptionGrid2 {
            id: optionGrid1
        }

    }

    Column {
        id: tableDisplay
        height: parent.height / 1.5
        width: parent.width
        anchors.top: queryElements.bottom
        anchors.left: parent.parent.left
        anchors.leftMargin: 25
        anchors.topMargin: 50

        spacing: 20

        TextArea {
            id: resultsTable1
            textFormat: Text.RichText
            wrapMode: TextEdit.Wrap
            anchors.right: parent.right
            width: parent.width
            height: parent.height / 3
            textColor: "#000000"

            function set_text(txt){
                resultsTable1.text = txt
            }

        }



       }

    TextArea {
        id: log
        objectName: "log"

        width: parent.width
        height: parent.height / 6

        wrapMode: TextEdit.Wrap
        textFormat: Text.RichText

        anchors.bottom: compute.top
        anchors.bottomMargin: 100

        function set_text(txt){
            resultsTable2.text = txt
        }
    }

    Text {
        id: txt2
        x: 5
        y: -2
        color: "#000000"
        text: "experimental conditions"
        font.bold: true
        anchors.horizontalCenterOffset: -456
        font.pointSize: 16
        horizontalAlignment: Text.AlignLeft
        anchors.topMargin: 52
        font.family: "Helvetica"
        style: Text.Raised
        anchors.top: parent.top
        textFormat: Text.AutoText
        anchors.horizontalCenter: parent.horizontalCenter
        styleColor: "#797979"
    }

}
