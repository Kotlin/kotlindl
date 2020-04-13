package tf_api.blocks.layers

import tf_api.blocks.Activation
import tf_api.blocks.Initializer

class Dense<T : Number>(
    outputSize: Int,
    activation: Activation = Activation.Relu,
    kernelInitializer: Initializer = Initializer.TRUNCATED_NORMAL,
    biasInitializer: Initializer = Initializer.ZEROS
) : Layer<T>() {
}