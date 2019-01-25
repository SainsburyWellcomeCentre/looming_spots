import QtQuick 2.0
import QtQuick.Controls 1.3

ComboBox {
    id: cmb3
    textRole: "key"

    function reload() {
        lm.reload();
    }

    model:  ListModel {
        id: lm
        function reload() {
            lm.clear()
            for (var i=0; i < py_iface.get_test_specific_options(); i++) {
                var value = py_iface.get_test_specific_options(i);
                lm.append({key: value});
            }
        }

        Component.onCompleted: {
            reload();

        }


    }

}
