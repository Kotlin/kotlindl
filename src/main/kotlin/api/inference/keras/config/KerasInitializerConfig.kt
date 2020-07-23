package api.inference.keras.config

data class KerasInitializerConfig(
    val distribution: String? = null,
    val maxval: Double? = null,
    val mean: Double? = null,
    val minval: Double? = null,
    val mode: String? = null,
    val scale: Double? = null,
    val seed: Int? = null,
    val stddev: Double? = null,
    val value: Int? = null
)