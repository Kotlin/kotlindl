package api.inference.keras.config

data class KerasSequentialModel(
    val backend: String? = "tensorflow",
    val class_name: String? = "Sequential",
    val config: SequentialConfig?,
    val keras_version: String? = "2.2.4-tf"
)