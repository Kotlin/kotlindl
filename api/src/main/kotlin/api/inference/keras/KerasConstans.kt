package api.inference.keras

// Keras layers
const val LAYER_CONV2D = "Conv2D"
const val LAYER_DENSE = "Dense"
const val LAYER_MAX_POOLING_2D = "MaxPooling2D"
const val LAYER_AVG_POOLING_2D = "AvgPooling2D"
const val LAYER_FLATTEN = "Flatten"

// Keras data types
const val DATATYPE_FLOAT32 = "float32"

// Keras Initializers
const val INITIALIZER_GLOROT_UNIFORM = "GlorotUniform"
const val INITIALIZER_GLOROT_NORMAL = "GlorotNormal"
const val INITIALIZER_HE_UNIFORM = "HeUniform"
const val INITIALIZER_HE_NORMAL = "HeNormal"
const val INITIALIZER_LECUN_UNIFORM = "LeCunUniform"
const val INITIALIZER_LECUN_NORMAL = "LeCunNormal"
const val INITIALIZER_ZEROS = "Zeros"
const val INITIALIZER_ONES = "Ones"
const val INITIALIZER_RANDOM_NORMAL = "RandomNormal"
const val INITIALIZER_RANDOM_UNIFORM = "RandomUniform"
const val INITIALIZER_TRUNCATED_NORMAL = "TruncatedNormal"
const val INITIALIZER_CONSTANT = "Constant"
const val INITIALIZER_VARIANCE_SCALING = "VarianceScaling"

// Keras activations
const val ACTIVATION_RELU = "relu"
const val ACTIVATION_SIGMOID = "sigmoid"
const val ACTIVATION_SOFTMAX = "softmax"
const val LINEAR = "linear"

// Layer settings
const val CHANNELS_LAST = "channels_last"
const val PADDING_SAME = "same"