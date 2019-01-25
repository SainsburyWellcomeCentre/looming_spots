import QtQuick 2.5
import QtQuick.Controls 1.3

Row{
    id: cmbRow
    width: 375
    height: 50
    
    function get_key() {
        return cmb.currentText
    }

    function get_value() {
        return cmb3.currentText
    }

    function get_comparator() {
        return cmb2.currentText
    }


    function reload() {
        cmb.reload()
        cmb2.reload()
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


        onCurrentIndexChanged: {
            console.log(cmb.currentText)
            py_iface.set_current_key(cmb.currentText)
            cmb3.reload()
        }
    }

    ComboBox {
        id: cmb2
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

    ComboBox {
        id: cmb3
        textRole: "key"

        function reload() {
            lm3.reload();
        }

        model:  ListModel {
            id: lm3
            function reload() {
                lm3.clear()
                for (var i=0; i < py_iface.get_n_options(); i++) {
                    var value = py_iface.get_option_at(i);
                    lm3.append({key: value});
                }
            }

            Component.onCompleted: {
                reload();
            }


        }

    }


}
