import QtQuick 2.5
import QtQuick.Controls 1.3


Column{
    id: queryElements
    anchors.top: parent.top
    anchors.left: parent.left
    anchors.topMargin: 130
    anchors.leftMargin: 20
    width: 350
    height: 184
    rotation: 0
    spacing: 2

    function get_keys(dict_idx) {
        for (var i = 0; i < children.length; i++)
        {
            py_iface.update_condition_dictionary(dict_idx, children[i].get_key(), children[i].get_value(), children[i].get_comparator());

        }
    }

    OptionGrid2 {
        id: optionGrid1
    }

    OptionGrid2 {
        id: optionGrid2
    }

    OptionGrid2 {
        id: optionGrid3
    }

    OptionGrid2 {
        id: optionGrid4
    }
}
