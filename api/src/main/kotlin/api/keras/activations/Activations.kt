package api.keras.activations

import api.getDType
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
    SoftSign,
    HardSigmoid,
    Swish;

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
                HardSigmoid -> HardSigmoidActivation()
                Swish -> SwishActivation()
            }
        }
    }
}


class LinearActivation<T : Number>() : Activation<T> {
    override fun apply(tf: Ops, features: Operand<T>, name: String): Operand<T> {
        return if (name.isNotEmpty()) {
            features
        } else {
            tf.withName("Activation_$name").identity(features)
        }
    }
}

class SigmoidActivation<T : Number>() : Activation<T> {
    override fun apply(tf: Ops, features: Operand<T>, name: String): Operand<T> {
        return if (name.isNotEmpty()) {
            tf.math.sigmoid(features)
        } else {
            tf.withName("Activation_$name").math.sigmoid(features)
        }
    }
}

class ReluActivation<T : Number>() : Activation<T> {
    override fun apply(tf: Ops, features: Operand<T>, name: String): Operand<T> {
        return if (name.isEmpty()) {
            tf.nn.relu(features)
        } else {
            tf.withName("Activation_$name").nn.relu(features)
        }
    }
}

class Relu6Activation<T : Number>() : Activation<T> {
    override fun apply(tf: Ops, features: Operand<T>, name: String): Operand<T> {
        return if (name.isNotEmpty()) {
            tf.nn.relu6(features)
        } else {
            tf.withName("Activation_$name").nn.relu6(features)
        }
    }
}

class TanhActivation<T : Number>() : Activation<T> {
    override fun apply(tf: Ops, features: Operand<T>, name: String): Operand<T> {
        return if (name.isNotEmpty()) {
            tf.math.tanh(features)
        } else {
            tf.withName("Activation_$name").math.tanh(features)
        }
    }
}

class EluActivation<T : Number>() : Activation<T> {
    override fun apply(tf: Ops, features: Operand<T>, name: String): Operand<T> {
        return if (name.isNotEmpty()) {
            tf.nn.elu(features)
        } else {
            tf.withName("Activation_$name").nn.elu(features)
        }
    }
}

class SeluActivation<T : Number>() : Activation<T> {
    override fun apply(tf: Ops, features: Operand<T>, name: String): Operand<T> {
        return if (name.isNotEmpty()) {
            tf.nn.selu(features)
        } else {
            tf.withName("Activation_$name").nn.selu(features)
        }
    }
}

class SoftmaxActivation<T : Number>() : Activation<T> {
    override fun apply(tf: Ops, features: Operand<T>, name: String): Operand<T> {
        return if (name.isNotEmpty()) {
            tf.nn.softmax(features)
        } else {
            tf.withName("Activation_$name").nn.softmax(features)
        }
    }
}

class LogSoftmaxActivation<T : Number>() : Activation<T> {
    override fun apply(tf: Ops, features: Operand<T>, name: String): Operand<T> {
        return tf.nn.logSoftmax(features)
    }
}

class ExponentialActivation<T : Number>() : Activation<T> {
    override fun apply(tf: Ops, features: Operand<T>, name: String): Operand<T> {
        return tf.math.exp(features)
    }
}

class SoftPlusActivation<T : Number>() : Activation<T> {
    override fun apply(tf: Ops, features: Operand<T>, name: String): Operand<T> {
        val one: Operand<T> = tf.dtypes.cast(tf.constant(1), getDType())

        return tf.math.log(tf.math.add(tf.math.exp(features), one))
    }
}

class SoftSignActivation<T : Number>() : Activation<T> {
    override fun apply(tf: Ops, features: Operand<T>, name: String): Operand<T> {
        return tf.nn.softsign(features)
    }
}

class HardSigmoidActivation<T : Number>() : Activation<T> {
    override fun apply(tf: Ops, features: Operand<T>, name: String): Operand<T> {
        val point2: Operand<T> = tf.dtypes.cast(tf.constant(0.2), getDType())
        val point5: Operand<T> = tf.dtypes.cast(tf.constant(0.5), getDType())

        return tf.math.add(tf.math.mul(features, point2), point5)
    }
}

class SwishActivation<T : Number>() : Activation<T> {
    override fun apply(tf: Ops, features: Operand<T>, name: String): Operand<T> {
        return tf.math.mul(features, tf.math.sigmoid(features))
    }
}