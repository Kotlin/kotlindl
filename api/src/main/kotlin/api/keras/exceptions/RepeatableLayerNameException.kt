package api.keras.exceptions

class RepeatableLayerNameException(layerName: String) :
    Exception("The layer name $layerName is used in previous layers. The layer name should be unique.") {
}
