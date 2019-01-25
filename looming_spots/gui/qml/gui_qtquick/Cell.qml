import QtQuick 2.0

Item {
    id: container
    property alias cellColor: rectangle.color
    signal clicked(color cellColor)

    width: 40; height: 25
}
