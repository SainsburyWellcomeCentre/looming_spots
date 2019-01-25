import QtQuick 2.5
import QtQuick.Controls 1.3

Row{

    width: 100
    height: 50

    function get_key() {
        return cmb.currentText
    }

    function get_value() {
        return txt.text
    }

    function get_comparator() {
        return comparatorCmb.currentText
    }


    function reload() {
        cmb.reload()
        comparatorCmb.reload()
    }

    ComboBox {
        id: cmb
        textRole: "key"

        function reload() {
            lm.reload();
        }

        model:  ListModel {
            id: lm
            function reload() {
                for (var i=0; i < py_iface.get_n_keys(); i++) {
                    var value = py_iface.get_key_at(i);
                    lm.append({key: value});
                }
            }

            Component.onCompleted: {
                reload();
            }
        }
    }

    TextField {
        id: txt
        placeholderText: qsTr("Text Field")
        width: parent.width
    }

    ComboBox {
        id: comparatorCmb
        textRole: "key"

        function reload() {
            lm2.reload();
        }

        model:  ListModel {
            id: lm2
            function reload() {
                for (var i=0; i < py_iface.get_n_comparators(); i++) {
                    var value = py_iface.get_comparator_at(i);
                    lm2.append({key: value});
                }
            }

            Component.onCompleted: {
                reload();
            }
        }
    }

}
