/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.activation

import org.tensorflow.Operand
import org.tensorflow.op.Ops

/**
 * Neural network hyperparameter, activation function of a node defines the output of that node given an input or set of inputs.
 */
public enum class Activations {
    /**
     * Linear unit. Returns unmodified input.
     *
     * NOTE: Doing nothing useful. Returns to ancient times of linear perceptron.
     *
     * Calls [LinearActivation] under the hood.
     */
    Linear,

    /**
     * Sigmoid activation function.
     *
     * Transforms input 'x' according formula:
     * ```
     * sigmoid(x) = 1 / (1 + exp(-x))
     * ```
     *
     * For small values (<-5), `sigmoid` returns a value close to zero, and for large values (>5)
     * the result of the function gets close to 1.
     *
     * NOTE: Sigmoid is equivalent to a 2-element Softmax, where the second element is
     * assumed to be zero. The sigmoid function always returns a value between 0 and 1.
     *
     * Calls [SigmoidActivation] under the hood.
     *
     */
    Sigmoid,

    /**
     * Hyperbolic tangent activation function.
     *
     * Transforms input 'x' according formula:
     * ```
     * tanh(x) = sinh(x)/cosh(x) = ((exp(x) - exp(-x))/(exp(x) + exp(-x)))
     * ```
     *
     * Calls [TanhActivation] under the hood.
     */
    Tanh,

    /**
     * Rectified linear unit (ReLU).
     *
     * With default values, this returns the standard ReLU activation:
     * `max(x, 0)`, the element-wise maximum of 0 and the input tensor.
     *
     * Calls [ReluActivation] under the hood.
     */
    Relu,

    /**
     * Computes Rectified Linear 6:
     * ```
     * min(max(features, 0), 6)
     * ```
     * Calls [Relu6Activation] under the hood.
     *
     * @see <a href="http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf">
     *     Convolutional Deep Belief Networks on CIFAR-10. A. Krizhevsky</a>
     */
    Relu6,

    /**
     * Exponential Linear Unit.
     *
     * The exponential linear unit (ELU) with `alpha > 0` is:
     * `x` if `x > 0` and `alpha * (exp(x) - 1)` if `x < 0`
     *
     * For this implementations alpha is equal to 1.0.
     *
     * The ELU hyperparameter `alpha` controls the value to which an
     * ELU saturates for negative net inputs. ELUs diminish the
     * vanishing gradient effect.
     *
     * ELUs have negative values which pushes the mean of the activations closer to zero.
     *
     * Mean activations that are closer to zero enable faster learning as they
     * bring the gradient closer to the natural gradient.
     *
     * ELUs saturate to a negative value when the argument gets smaller.
     * Saturation means a small derivative which decreases the variation
     * and the information that is propagated to the next layer.
     *
     * Calls [EluActivation] under the hood.
     *
     * @see <a href="https://arxiv.org/abs/1511.07289">Fast and Accurate Deep Network Learning by Exponential Linear Units
     * (ELUs) (Clevert et al, 2016)</a>
     */
    Elu,

    /**
     * Scaled Exponential Linear Unit (SELU).
     *
     * The Scaled Exponential Linear Unit (SELU) activation function is defined as:
     * ```
     *  if x > 0: return scale * x
     *  if x < 0: return scale * alpha * (exp(x) - 1)
     * ```
     * where `alpha` and `scale` are pre-defined constants (`alpha=1.67326324` and `scale=1.05070098`).
     *
     * Basically, the SELU activation function multiplies `scale` (> 1) with the
     * output of the `tf.keras.activations.elu` function to ensure a slope larger
     * than one for positive inputs.
     *
     * NOTE: The values of `alpha` and `scale` are
     * chosen so that the mean and variance of the inputs are preserved
     * between two consecutive layers as long as the weights are initialized
     * correctly (see [org.jetbrains.kotlinx.dl.api.core.initializer.LeCunNormal] initializer)
     * and the number of input units is "large enough"
     * (see reference paper for more information).
     *
     * Calls [SeluActivation] under the hood.
     *
     * @see <a href="https://arxiv.org/abs/1706.02515">Klambauer et al., 2017</a>
     */
    Selu,

    /**
     * Softmax converts a real vector to a vector of categorical probabilities.
     * The elements of the output vector are in range (0, 1) and sum to 1.
     *
     * Softmax is often used as the activation for the last
     * layer of a classification network because the result could be interpreted as
     * a probability distribution.
     *
     * Calls [SoftmaxActivation] under the hood.
     */
    Softmax,

    /**
     *
     */
    LogSoftmax,

    /**
     * Exponential activation function.
     *
     * Transforms input 'x' according formula:
     * ```
     * exp(x)
     * ```
     *
     * Calls [ExponentialActivation] under the hood.
     */
    Exponential,

    /**
     * Softplus activation function.
     *
     * Transforms input 'x' according formula:
     * ```
     * softplus(x) = log(exp(x) + 1)
     * ```
     *
     * Calls [SoftPlusActivation] under the hood.
     */
    SoftPlus,

    /***
     * Softsign activation function.
     *
     * Transforms input 'x' according formula:
     * ```
     * softsign(x) = x / (abs(x) + 1)
     * ```
     *
     * Calls [SoftSignActivation] under the hood.
     */
    SoftSign,

    /**
     * Hard sigmoid activation function.
     *
     * Transforms input 'x' according formula:
     * ```
     * if x < -2.5: return 0
     * if x > 2.5: return 1
     * if -2.5 <= x <= 2.5: return 0.2 * x + 0.5
     * ```
     * A faster approximation of the sigmoid activation.
     *
     * Calls [HardSigmoidActivation] under the hood.
     */
    HardSigmoid,

    /**
     * Swish activation function.
     *
     * Transforms input 'x' according formula:
     * ```
     * swish(x) = x * sigmoid(x)
     * ```
     *
     * It is a smooth, non-monotonic function that consistently matches
     * or outperforms ReLU on deep networks, it is unbounded above and
     * bounded below.
     *
     * Calls [SwishActivation] under the hood.
     *
     * @see <a href="https://arxiv.org/abs/1710.05941">Ramachandran et al., 2017</a>
     */
    Swish,

    /**
     * Mish activation function.
     *
     * Transforms input 'x' according formula:
     * ```
     * mish(x) = x * tanh(softplus(x))
     * ```
     *
     * It is a smooth, non-monotonic function that consistently matches
     * or outperforms ReLU and Swish on deep networks, it is unbounded above and
     * bounded below. It also smoothens the loss landscape of the network.
     *
     * Calls [MishActivation] under the hood.
     *
     * @see <a href="https://arxiv.org/abs/1908.08681">Misra, 2019</a>
     */
    Mish,

    /**
     * HardShrink Function
     *
     * Computes hard shrink function:
     *
     * hardshrink(x) = x if x < lower
     *                 x if x > upper
     *                 0 otherwise
     *
     * Calls [HardShrinkActivation] under the hood.
     * @property [lower] lower bound for setting values to zeros
     * @property [upper] upper bound for setting values to zeros
     */
    HardShrink,

    /**
     * Non-Parametric Linearly Scaled Hyperbolic Tangent (LiSHT) Activation Function.
     *
     * ```
     * LiSHT(x) = x * tanh(x)
     * ```
     */
    LiSHT,

    /**
     * Snake Activation Function.
     *
     * Transforms input 'x' according formula:
     * ```
     * snake(x) = x + (1 - cos(2 * frequency * x)) / (2 * frequency)
     * ```
     * See [Neural Networks Fail to Learn Periodic Functions and How to Fix It](https://arxiv.org/abs/2006.08195).
     * @property [frequency] A scalar, frequency of the periodic part
     */

    Snake;

    public companion object {
        /**
         * Converts [activationType] to the appropriate [Activation] sub-class.
         */
        public fun convert(activationType: Activations): Activation {
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
                Mish -> MishActivation()
                HardShrink -> HardShrinkActivation()
                LiSHT -> LishtActivation()
                Snake -> SnakeActivation()
            }
        }
    }
}

/**
 * @see [Activations.Linear]
 */
public class LinearActivation : Activation {
    override fun apply(tf: Ops, features: Operand<Float>): Operand<Float> {
        return features
    }
}

/**
 * @see [Activations.Sigmoid]
 */
public class SigmoidActivation : Activation {
    override fun apply(tf: Ops, features: Operand<Float>): Operand<Float> = tf.math.sigmoid(features)
}

/**
 * @see [Activations.Relu]
 */
public class ReluActivation : Activation {
    override fun apply(tf: Ops, features: Operand<Float>): Operand<Float> = tf.nn.relu(features)
}

/**
 * @see [Activations.Relu6]
 */
public class Relu6Activation : Activation {
    override fun apply(tf: Ops, features: Operand<Float>): Operand<Float> = tf.nn.relu6(features)
}

/**
 * @see [Activations.Tanh]
 */
public class TanhActivation : Activation {
    override fun apply(tf: Ops, features: Operand<Float>): Operand<Float> = tf.math.tanh(features)
}

/**
 * @see [Activations.Elu]
 */
public class EluActivation : Activation {
    override fun apply(tf: Ops, features: Operand<Float>): Operand<Float> = tf.nn.elu(features)
}

/**
 * @see [Activations.Selu]
 */
public class SeluActivation : Activation {
    override fun apply(tf: Ops, features: Operand<Float>): Operand<Float> = tf.nn.selu(features)
}

/**
 * Internal class, wrapping TensorFlow operand
 * ```
 * tf.nn.softmax
 * ```
 *
 * For each batch `i` and class `j` we have
 *
 * ```
 * softmax[i, j] = exp(logits[i, j]) / sum_j(exp(logits[i, j]))
 * ```
 *
 * @see [Activations.Softmax] for explanation.
 */
public class SoftmaxActivation : Activation {
    override fun apply(tf: Ops, features: Operand<Float>): Operand<Float> = tf.nn.softmax(features)
}

/**
 * @see [Activations.LogSoftmax]
 */
public class LogSoftmaxActivation : Activation {
    override fun apply(tf: Ops, features: Operand<Float>): Operand<Float> = tf.nn.logSoftmax(features)
}

/**
 * @see [Activations.Exponential]
 */
public class ExponentialActivation : Activation {
    override fun apply(tf: Ops, features: Operand<Float>): Operand<Float> = tf.math.exp(features)
}

/**
 * @see [Activations.SoftPlus]
 */
public class SoftPlusActivation : Activation {
    override fun apply(tf: Ops, features: Operand<Float>): Operand<Float> =
        tf.math.log(tf.math.add(tf.math.exp(features), tf.constant(1.0f)))
}

/**
 * @see [Activations.SoftSign]
 */
public class SoftSignActivation : Activation {
    override fun apply(tf: Ops, features: Operand<Float>): Operand<Float> = tf.nn.softsign(features)
}

/**
 * @see [Activations.HardSigmoid]
 */
public class HardSigmoidActivation : Activation {
    override fun apply(tf: Ops, features: Operand<Float>): Operand<Float> {
        val point2: Operand<Float> = tf.constant(0.2f)
        val point5: Operand<Float> = tf.constant(0.5f)

        return tf.math.add(tf.math.mul(features, point2), point5)
    }
}

/**
 * @see [Activations.Swish]
 */
public class SwishActivation : Activation {
    override fun apply(tf: Ops, features: Operand<Float>): Operand<Float> =
        tf.math.mul(features, tf.math.sigmoid(features))
}

/**
 * @see [Activations.Mish]
 */
public class MishActivation : Activation {
    override fun apply(tf: Ops, features: Operand<Float>): Operand<Float> =
        tf.math.mul(features, tf.math.tanh(tf.math.softplus(features)))
}

/**
 * @see [Activations.HardShrink]
 */
public class HardShrinkActivation(public val lower: Float = -0.5f, public val upper: Float = 0.5f) : Activation {
    override fun apply(tf: Ops, features: Operand<Float>): Operand<Float> {
        require(lower < upper) {
            "The value of lower should not be higher than upper"
        }
        val maskLower = tf.math.minimum(features, tf.constant(lower)) != tf.constant(lower)
        val maskUpper = tf.math.maximum(features, tf.constant(upper)) != tf.constant(upper)
        val mask = (maskLower || maskUpper)
        return when (mask) {
            false -> tf.constant(0) as Operand<Float>
            true -> features
        }
    }
}

/**
 * @see [Activations.LiSHT]
 */
public class LishtActivation : Activation {
    override fun apply(tf: Ops, features: Operand<Float>): Operand<Float> =
        tf.math.mul(features, tf.math.tanh(features))
}


/**
* @see [Activations.Snake]
*/

public class SnakeActivation(private val frequency: Float = 1.0f) : Activation {
    override fun apply(tf: Ops, features: Operand<Float>): Operand<Float> {
        val f = tf.constant(frequency)
        val fDouble = tf.math.mul(tf.constant(2.0f), f) // returns 2 * frequency

        return tf.math.add(features,
            tf.math.div(tf.math.sub(tf.constant(1.0f), tf.math.cos(tf.math.mul(fDouble, features))), fDouble))
    }
}

