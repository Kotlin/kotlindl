package api.inference.keras.config

data class ConfigX(
    val activation: String? = null,
    val activity_regularizer: ActivityRegularizer? = null,
    var batch_input_shape: List<Int?>? = null,
    val bias_constraint: Any? = null,
    val bias_initializer: BiasInitializer? = null,
    val bias_regularizer: BiasRegularizer? = null,
    val data_format: String? = null,
    val dilation_rate: List<Int>? = null,
    val dtype: String? = null,
    val filters: Int? = null,
    val kernel_constraint: Any? = null,
    val kernel_initializer: KernelInitializer? = null,
    val kernel_regularizer: KernelRegularizer? = null,
    val kernel_size: List<Int>? = null,
    val name: String? = null,
    val padding: String? = null,
    val pool_size: List<Int>? = null,
    val strides: List<Int>? = null,
    val trainable: Boolean? = true,
    val units: Int? = null,
    val use_bias: Boolean? = true
)