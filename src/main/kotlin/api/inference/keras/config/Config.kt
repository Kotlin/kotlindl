package api.inference.keras.config

data class Config(
    val layers: List<KerasLayer>?,
    val name: String?
)