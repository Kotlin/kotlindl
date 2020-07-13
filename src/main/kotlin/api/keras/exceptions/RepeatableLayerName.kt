package api.keras.exceptions

class RepeatableLayerName(layerName: String) :
    Exception("The layer name $layerName is used in previous layers. The layer name should be unique") {
}
