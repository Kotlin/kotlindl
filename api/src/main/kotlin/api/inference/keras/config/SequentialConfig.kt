package api.inference.keras.config

data class SequentialConfig(
    val layers: List<KerasLayer>?,
    val name: String?
)