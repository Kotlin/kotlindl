package api.keras.activations

import org.tensorflow.Operand
import org.tensorflow.op.Ops

enum class Activations {
    Linear,
    Sigmoid,
    Tanh,
    Relu,
    Relu6,
    Elu,
    Selu,
    Softmax,
    LogSoftmax,
    Exponential,
    SoftPlus,
    SoftSign;

    companion object {
        fun <T : Number> convert(activationType: Activations): Activation<T> {
            return when (activationType) {
                Sigmoid -> SigmoidActivation()
                Linear -> LinearActivation()
                Tanh -> TanhActivation()
                Relu -> ReluActivation()
                Relu6 -> Relu6Activation()
                Elu -> EluActivation()
                Selu -> SeluActivation()
                Softmax -> SoftmaxActivation()
                LogSoftmax -> LogSoftmaxActivation()
                Exponential -> ExponentialActivation()
                SoftPlus -> SoftPlusActivation()
                SoftSign -> SoftSignActivation()
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

class Relu6Activation<T : Number>() : Activation<T> {
    override fun apply(tf: Ops, features: Operand<T>): Operand<T> {
        return tf.nn.relu6(features)
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

class SeluActivation<T : Number>() : Activation<T> {
    override fun apply(tf: Ops, features: Operand<T>): Operand<T> {
        return tf.nn.selu(features)
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

class ExponentialActivation<T : Number>() : Activation<T> {
    override fun apply(tf: Ops, features: Operand<T>): Operand<T> {
        return tf.math.exp(features)
    }
}

class SoftPlusActivation<T : Number>() : Activation<T> {
    override fun apply(tf: Ops, features: Operand<T>): Operand<T> {
        return tf.math.log(tf.math.add(tf.math.exp(features), tf.constant(1) as Operand<T>))
    }
}

class SoftSignActivation<T : Number>() : Activation<T> {
    override fun apply(tf: Ops, features: Operand<T>): Operand<T> {
        return tf.nn.softsign(features)
    }
}