import QtQuick 2.5
import QtQuick.Controls 1.3

ComboBox {
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
