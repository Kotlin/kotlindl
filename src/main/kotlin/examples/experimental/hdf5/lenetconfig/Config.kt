package examples.experimental.hdf5.lenetconfig

data class Config(
    val layers: List<KerasLayer>?,
    val name: String?
)