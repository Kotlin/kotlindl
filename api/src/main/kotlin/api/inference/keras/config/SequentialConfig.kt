package api.inference.keras.config

internal data class SequentialConfig(
    val layers: List<KerasLayer>?,
    val name: String?
)