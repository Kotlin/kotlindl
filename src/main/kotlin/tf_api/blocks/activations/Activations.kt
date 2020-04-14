package tf_api.blocks.activations

import org.tensorflow.Operand
import org.tensorflow.op.Ops

enum class Activations {
    Linear,
    Sigmoid,
    Tanh,
    Relu,
    Elu,
    Selu,
    Softmax,
    LogSoftmax;

    companion object {
        fun <T : Number> convert(activationType: Activations): Activation<T> {
            return when (activationType) {
                Sigmoid -> SigmoidActivation<T>()
                Linear -> LinearActivation<T>()
                Tanh -> TanhActivation<T>()
                Relu -> ReluActivation<T>()
                Elu -> EluActivation<T>()
                Selu -> TODO()
                Softmax -> SoftmaxActivation<T>()
                LogSoftmax -> LogSoftmaxActivation<T>()
            }
        }
    }
}

class LinearActivation<T : Number>() : Activation<T> {
    override fun apply(tf: Ops, features: Operand<T>): Operand<T> {
        return features
    }
}

class SigmoidActivation<T : Number>() : Activation<T> {
    override fun apply(tf: Ops, features: Operand<T>): Operand<T> {
        return tf.math.sigmoid(features)
    }
}

class ReluActivation<T : Number>() : Activation<T> {
    override fun apply(tf: Ops, features: Operand<T>): Operand<T> {
        return tf.nn.relu(features)
    }
}

class TanhActivation<T : Number>() : Activation<T> {
    override fun apply(tf: Ops, features: Operand<T>): Operand<T> {
        return tf.math.tanh(features)
    }
}

class EluActivation<T : Number>() : Activation<T> {
    override fun apply(tf: Ops, features: Operand<T>): Operand<T> {
        return tf.nn.elu(features)
    }
}

class SoftmaxActivation<T : Number>() : Activation<T> {
    override fun apply(tf: Ops, features: Operand<T>): Operand<T> {
        return tf.nn.softmax(features)
    }
}

class LogSoftmaxActivation<T : Number>() : Activation<T> {
    override fun apply(tf: Ops, features: Operand<T>): Operand<T> {
        return tf.nn.logSoftmax(features)
    }
}