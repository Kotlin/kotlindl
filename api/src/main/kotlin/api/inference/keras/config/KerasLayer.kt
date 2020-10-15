package api.inference.keras.config

internal data class KerasLayer(
    val class_name: String?,
    val config: LayerConfig?
)