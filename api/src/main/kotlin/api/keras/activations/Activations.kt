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
        fun convert(activationType: Activations): Activation {
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


class LinearActivation : Activation {
    override fun apply(tf: Ops, features: Operand<Float>, name: String): Operand<Float> {
        return if (name.isNotEmpty()) {
            features
        } else {
            tf.withName("Activation_$name").identity(features)
        }
    }
}

class SigmoidActivation : Activation {
    override fun apply(tf: Ops, features: Operand<Float>, name: String): Operand<Float> {
        return if (name.isNotEmpty()) {
            tf.math.sigmoid(features)
        } else {
            tf.withName("Activation_$name").math.sigmoid(features)
        }
    }
}

class ReluActivation : Activation {
    override fun apply(tf: Ops, features: Operand<Float>, name: String): Operand<Float> {
        return if (name.isEmpty()) {
            tf.nn.relu(features)
        } else {
            tf.withName("Activation_$name").nn.relu(features)
        }
    }
}

class Relu6Activation : Activation {
    override fun apply(tf: Ops, features: Operand<Float>, name: String): Operand<Float> {
        return if (name.isNotEmpty()) {
            tf.nn.relu6(features)
        } else {
            tf.withName("Activation_$name").nn.relu6(features)
        }
    }
}

class TanhActivation : Activation {
    override fun apply(tf: Ops, features: Operand<Float>, name: String): Operand<Float> {
        return if (name.isNotEmpty()) {
            tf.math.tanh(features)
        } else {
            tf.withName("Activation_$name").math.tanh(features)
        }
    }
}

class EluActivation : Activation {
    override fun apply(tf: Ops, features: Operand<Float>, name: String): Operand<Float> {
        return if (name.isNotEmpty()) {
            tf.nn.elu(features)
        } else {
            tf.withName("Activation_$name").nn.elu(features)
        }
    }
}

class SeluActivation : Activation {
    override fun apply(tf: Ops, features: Operand<Float>, name: String): Operand<Float> {
        return if (name.isNotEmpty()) {
            tf.nn.selu(features)
        } else {
            tf.withName("Activation_$name").nn.selu(features)
        }
    }
}

class SoftmaxActivation : Activation {
    override fun apply(tf: Ops, features: Operand<Float>, name: String): Operand<Float> {
        return if (name.isNotEmpty()) {
            tf.nn.softmax(features)
        } else {
            tf.withName("Activation_$name").nn.softmax(features)
        }
    }
}

class LogSoftmaxActivation : Activation {
    override fun apply(tf: Ops, features: Operand<Float>, name: String): Operand<Float> {
        return tf.nn.logSoftmax(features)
    }
}

class ExponentialActivation : Activation {
    override fun apply(tf: Ops, features: Operand<Float>, name: String): Operand<Float> {
        return tf.math.exp(features)
    }
}

class SoftPlusActivation : Activation {
    override fun apply(tf: Ops, features: Operand<Float>, name: String): Operand<Float> {
        val one: Operand<Float> = tf.dtypes.cast(tf.constant(1), getDType())

        return tf.math.log(tf.math.add(tf.math.exp(features), one))
    }
}

class SoftSignActivation : Activation {
    override fun apply(tf: Ops, features: Operand<Float>, name: String): Operand<Float> {
        return tf.nn.softsign(features)
    }
}

class HardSigmoidActivation : Activation {
    override fun apply(tf: Ops, features: Operand<Float>, name: String): Operand<Float> {
        val point2: Operand<Float> = tf.dtypes.cast(tf.constant(0.2), getDType())
        val point5: Operand<Float> = tf.dtypes.cast(tf.constant(0.5), getDType())

        return tf.math.add(tf.math.mul(features, point2), point5)
    }
}

class SwishActivation : Activation {
    override fun apply(tf: Ops, features: Operand<Float>, name: String): Operand<Float> {
        return tf.math.mul(features, tf.math.sigmoid(features))
    }
}