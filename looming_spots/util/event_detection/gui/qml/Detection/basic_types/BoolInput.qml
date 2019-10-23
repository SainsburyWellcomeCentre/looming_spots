import QtQuick 2.5
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.3

//import "../style"

Item {
    id: root
    height: 25
    width: 130

    property alias label: label.text
    property alias tooltip: label.help

    property alias checked: checkBox.checked
    property alias boxWidth: checkBox.width

    signal gotChecked()

    Row {
        anchors.fill: parent
        spacing: width - (label.width + checkBox.width)

        LabelWTooltip {
            id: label
//            width: (parent.width -5) /2
            width: contentWidth + 5
            height: parent.height
            text: "Label"
        }
        CheckBox {
            id: checkBox
            onCheckedChanged: {
                root.gotChecked();
            }
        }
    }
}
