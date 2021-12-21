/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.activation

import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Stack

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
     * For this implementation alpha is equal to 1.0.
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
     * (ELUs) (Clevert et al., 2016)</a>
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
     * Log softmax activation function.
     *
     *  For each batch `i` and class `j` we have
     *  ```
     *  logsoftmax = logits - log(reduce_sum(exp(logits), axis))
     *  ```
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
     * Softshrink activation function.
     *
     * Transforms input 'x' according formula:
     * ```
     * if x > lambda: return x − lambda
     * if x < -lambda: return x + lambda
     * otherwise return 0
     * ```
     * A faster approximation of the sigmoid activation.
     *
     * Calls [SoftShrinkActivation] under the hood.
     */
    SoftShrink,

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
     */
    HardShrink,

    /**
     * Gelu Function
     *
     * Computes the Gaussian Error Linear Unit (GELU):
     *
     * gelu(x) = x * P(X <= x) where P(X) ~ N(0, 1)
     *
     * Calls [GeluActivation] under the hood.
     */
    Gelu,

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
     * @see [Neural Networks Fail to Learn Periodic Functions and How to Fix It](https://arxiv.org/abs/2006.08195).
     */
    Snake,

    /**
     * TanhShrink Activation Function.
     *
     * This is a hyperbolic tangent (TanH) shrink activation type that implements the element wise function:
     * ```
     * TanhShrink(x) = x − tanh(x)
     * ```
     * Calls [TanhActivation] under the hood.
     */
    TanhShrink,

    /**
     * Sparsemax activation function is similar to softmax but able to output sparse probabilities.
     *
     * for batch `i` and class `j`
     *
     * sparsemax(x)`[i,j]` = max(0, logits`[i,j]` - `τ`(logits`[i,:]`))
     *
     * @see <a href="https://arxiv.org/abs/1602.02068">From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification</a>
     */
    Sparsemax;

    public companion object {
        /**
         * Converts [activationType] to the appropriate [Activation] sub-class.
         */
        public fun convert(activationType: Activations): Activation {
            return when (activationType) {
                Sigmoid -> SigmoidActivation()
                Linear -> LinearActivation()
                Tanh -> TanhActivation()
                TanhShrink -> TanhShrinkActivation()
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
                SoftShrink -> SoftShrinkActivation()
                LiSHT -> LishtActivation()
                Snake -> SnakeActivation()
                Gelu -> GeluActivation()
                Sparsemax -> SparsemaxActivation()
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
 * @see [Activations.TanhShrink]
 */
public class TanhShrinkActivation : Activation {
    override fun apply(tf: Ops, features: Operand<Float>): Operand<Float> =
        tf.math.sub(features, tf.math.tanh(features))
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
 * @property [lower] lower bound for setting values to zeros
 * @property [upper] upper bound for setting values to zeros
 *
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
 * @property [lower] lower bound for setting values to zeros
 * @property [upper] upper bound for setting values to zeros

 * @see [Activations.SoftShrink]
 */
public class SoftShrinkActivation(public val lower: Float = -0.5f, public val upper: Float = 0.5f) : Activation {
    override fun apply(tf: Ops, features: Operand<Float>): Operand<Float> {
        require((lower < upper) && (lower < 0) && (upper > 0)) {
            "The boundary values have to be non zero and the lower bound has to be lower as the upper"
        }
        val zeros = tf.math.mul(features, tf.constant(0f))
        val valuesBelowLower = tf.where3(
            tf.math.less(features, tf.constant(lower)),
            tf.math.sub(
                features, tf.constant(lower)
            ),
            zeros
        )
        val valuesAboveUpper = tf.where3(
            tf.math.less(tf.constant(upper), features),
            tf.math.sub(
                features, tf.constant(upper)
            ),
            zeros
        )
        return tf.math.add(valuesBelowLower, valuesAboveUpper)
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
 * @property [frequency] A scalar, frequency of the periodic part.
 * @see [Activations.Snake]
 */
public class SnakeActivation(private val frequency: Float = 1.0f) : Activation {
    override fun apply(tf: Ops, features: Operand<Float>): Operand<Float> {
        val doubleFreqConstant = tf.constant(2 * frequency)

        return tf.math.add(
            features,
            tf.math.div(
                tf.math.sub(tf.constant(1.0f), tf.math.cos(tf.math.mul(doubleFreqConstant, features))),
                doubleFreqConstant
            )
        )
    }
}

/**
 * @property [approximate] The boolean flag to toggle approximation.
 *
 * @see [Activations.Gelu]
 */
public class GeluActivation(public val approximate: Boolean = false) : Activation {
    override fun apply(tf: Ops, features: Operand<Float>): Operand<Float> {
        if (approximate) {
            val coefficient = tf.constant(0.044715f)
            return tf.math.mul(
                tf.constant(0.5f), tf.math.mul(
                    features, tf.math.add(
                        tf.constant(1.0f), tf.math.tanh(
                            tf.math.mul(
                                tf.constant(0.7978845608028654f),       // This value is equal to sqrt(2/pi) to avoid a constant division
                                tf.math.add(features, tf.math.mul(coefficient, tf.math.pow(features, tf.constant(3f))))
                            )
                        )
                    )
                )
            )
        } else {
            return tf.math.mul(
                tf.constant(0.5f),
                tf.math.mul(
                    features,
                    tf.math.add(
                        tf.constant(1f),
                        tf.math.erf(tf.math.div(features, tf.constant(1.4142135623730951f)))
                    )
                )
            )
        }
    }
}

/**
 * @property [axis] axis along which the sparsemax operation is applied.
 * @see [Activations.Sparsemax]
 */
public class SparsemaxActivation(private val axis: Int = -1) : Activation {
    override fun apply(tf: Ops, features: Operand<Float>): Operand<Float> {

        // Keep references to shape because we perform sparsemax on 2D.
        // If required, we need to reshape features to 2D and back.
        val shape = features.asOutput().shape()
        val rank = shape.numDimensions()

        val isLastAxis = (axis == -1) || (axis == rank - 1)
        if (isLastAxis) {
            val output = compute2DSparsemax(tf, features)
            return tf.ensureShape(output, shape)
        }

        // compute2DSparsemax only calculates sparsemax operation along it's last axis.
        // If different axis is required for sparsemax, we first swap axes, then calculate compute2DSparsemax then
        // swap axes back.
        val axisNorm = axis % rank //ensure axis is within rank
        val logits = swapAxis(tf, features, axisNorm, rank - 1)

        val output = compute2DSparsemax(tf, logits)
        return tf.ensureShape(swapAxis(tf, output, axisNorm, rank - 1), shape)
    }

    private fun swapAxis(tf: Ops, features: Operand<Float>, axis: Int, lastIndex: Int): Operand<Float> {
        /**
         * swaps features Operand's lastIndex with axis
         */

        val range = (tf.range(tf.constant(0), tf.constant(lastIndex + 1), tf.constant(1)))
        return tf.linalg.transpose(
            features,
            tf.tensorScatterUpdate(
                range,
                tf.constant(arrayOf(intArrayOf(axis), intArrayOf(lastIndex))),
                tf.constant(intArrayOf(lastIndex, axis))
            )
        )
    }

    private fun compute2DSparsemax(tf: Ops, features: Operand<Float>): Operand<Float> {
        val shape = features.asOutput().tensor().shape()
        val dims = shape[shape.lastIndex]
        val dimsOp = tf.constant(dims.toInt())
        val obs = shape.reduce { acc, l -> acc * l } / dims
        val one = tf.constant(1f)

        val z = tf.reshape(features, tf.constant(longArrayOf(obs, dims)))
        val zSorted = tf.nn.topK(z, dimsOp)
        val zCumSum = tf.math.cumsum(zSorted.values(), tf.constant(-1))

        val k = tf.range(one, tf.math.add(tf.dtypes.cast(dimsOp, Float::class.javaObjectType), one), one)

        // check where (k * z_sorted + 1 > cumsum(z)
        val zCheck = tf.math.greater(tf.math.add(one, tf.math.mul(k, zSorted.values())), zCumSum)

        // casting boolean values to Int makes true = 1, false = 0,
        // then summing each row is same as finding last value that is one in such vector [1,1,..1,0,0,..,0]
        val kz = tf.reduceSum(tf.dtypes.cast(zCheck, Int::class.javaObjectType), tf.constant(-1))


        // If there are inf values or all values are -inf, the k_z will be zero,
        // this is mathematically invalid and will also cause the gather_nd to fail.
        // Prevent this issue for now by setting k_z = 1 if k_z = 0, this is then
        // fixed later (see p_safe) by returning p = nan. This results in the same
        // behavior as softmax.
        // (This comment is taken from original python implementation)
        val kzSafe = tf.math.maximum(kz, tf.constant(1))
        val indices = tf.stack(
            listOf(
                tf.range(tf.constant(0), tf.constant(obs.toInt()), tf.constant(1)),
                tf.math.sub(tf.reshape(kzSafe, tf.constant(intArrayOf(-1))), tf.constant(1))
            ), Stack.axis(1)
        )

        val tauSum = tf.gatherNd(zCumSum, indices)
        val tauZ = tf.math.div(tf.math.sub(tauSum, one), tf.dtypes.cast(kz, Float::class.javaObjectType))

        val p = tf.math.maximum(tf.constant(0f), tf.math.sub(z, tf.expandDims(tauZ, tf.constant(-1))))

        // getting a reference to last index. Similar to slicing  the last index of a python array [:, -1]
        val zCumsumLastIndex = tf.stack(
            listOf(
                tf.range(tf.constant(0), tf.constant(obs.toInt()), tf.constant(1)),
                tf.fill(tf.constant(longArrayOf(obs)), tf.math.sub(dimsOp, tf.constant(1)))
            ), Stack.axis(1)
        )

        val pSafe = tf.where3(
            tf.math.logicalOr(
                tf.math.equal(kz, tf.constant(0)),
                tf.math.isNan(
                    tf.gatherNd(zCumSum, zCumsumLastIndex)
                )
            ),
            tf.fill(tf.constant(longArrayOf(obs, dims)), tf.constant(Float.NaN)),
            p
        )

        return tf.reshape(pSafe, tf.constant(shape))
    }
}
