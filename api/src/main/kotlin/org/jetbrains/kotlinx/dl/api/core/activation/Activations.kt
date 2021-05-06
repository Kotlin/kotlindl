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
public object Activations {
    /**
     * Linear unit. Returns unmodified input.
     *
     * NOTE: Doing nothing useful. Returns to ancient times of linear perceptron.
     */
    public val Linear: Activation = activationLayer { _, features -> features }

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
     */
    public val Sigmoid: Activation = activationLayer { tf, features -> tf.math.sigmoid(features) }

    /**
     * Hyperbolic tangent activation function.
     *
     * Transforms input 'x' according formula:
     * ```
     * tanh(x) = sinh(x)/cosh(x) = ((exp(x) - exp(-x))/(exp(x) + exp(-x)))
     * ```
     */
    public val Tanh: Activation = activationLayer { tf, features -> tf.math.tanh(features) }

    /**
     * Rectified linear unit (ReLU).
     *
     * With default values, this returns the standard ReLU activation:
     * `max(x, 0)`, the element-wise maximum of 0 and the input tensor.
     */
    public val Relu: Activation = activationLayer { tf, features -> tf.nn.relu(features) }

    /**
     * Computes Rectified Linear 6:
     * ```
     * min(max(features, 0), 6)
     * ```
     *
     * @see <a href="http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf">
     *     Convolutional Deep Belief Networks on CIFAR-10. A. Krizhevsky</a>
     */
    public val Relu6: Activation = activationLayer { tf, features -> tf.nn.relu6(features) }

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
     * @see <a href="https://arxiv.org/abs/1511.07289">Fast and Accurate Deep Network Learning by Exponential Linear Units
     * (ELUs) (Clevert et al, 2016)</a>
     */
    public val Elu: Activation = activationLayer { tf, features -> tf.nn.elu(features) }

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
     * @see <a href="https://arxiv.org/abs/1706.02515">Klambauer et al., 2017</a>
     */
    public val Selu: Activation = activationLayer { tf, features -> tf.nn.selu(features) }

    /**
     * Softmax converts a real vector to a vector of categorical probabilities.
     * The elements of the output vector are in range (0, 1) and sum to 1.
     *
     * Softmax is often used as the activation for the last
     * layer of a classification network because the result could be interpreted as
     * a probability distribution.
     *
     * For each batch `i` and class `j` we have
     *
     * ```
     * softmax[i, j] = exp(logits[i, j]) / sum_j(exp(logits[i, j]))
     * ```
     */
    public val Softmax: Activation = activationLayer { tf, features -> tf.nn.softmax(features) }

    /**
     *
     */
    public val LogSoftmax: Activation = activationLayer { tf, features -> tf.nn.logSoftmax(features) }

    /**
     * Exponential activation function.
     *
     * Transforms input 'x' according formula:
     * ```
     * exp(x)
     * ```
     */
    public val Exponential: Activation = activationLayer { tf, features -> tf.math.exp(features) }

    /**
     * Softplus activation function.
     *
     * Transforms input 'x' according formula:
     * ```
     * softplus(x) = log(exp(x) + 1)
     * ```
     */
    public val SoftPlus: Activation = activationLayer { tf, features ->
        tf.math.log(tf.math.add(tf.math.exp(features), tf.constant(1.0f)))
    }

    /***
     * Softsign activation function.
     *
     * Transforms input 'x' according formula:
     * ```
     * softsign(x) = x / (abs(x) + 1)
     * ```
     */
    public val SoftSign: Activation = activationLayer { tf, features -> tf.nn.softsign(features) }

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
     */
    public val HardSigmoid: Activation = activationLayer { tf, features ->
        val point2: Operand<Float> = tf.constant(0.2f)
        val point5: Operand<Float> = tf.constant(0.5f)

        tf.math.add(tf.math.mul(features, point2), point5)
    }

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
     * @see <a href="https://arxiv.org/abs/1710.05941">Ramachandran et al., 2017</a>
     */
    public val Swish: Activation = activationLayer { tf, features -> tf.math.mul(features, tf.math.sigmoid(features)) }

    private inline fun activationLayer(crossinline operation: (Ops, Operand<Float>) -> Operand<Float>) =
        object : Activation {
            override fun apply(tf: Ops, features: Operand<Float>): Operand<Float> = operation(tf, features)
        }
}
