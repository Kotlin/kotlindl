/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.GlorotUniform
import org.jetbrains.kotlinx.dl.api.core.initializer.Initializer
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable


/**
 * units: Positive integer, dimensionality of the output space.
activation: Activation function to use.
Default: hyperbolic tangent (`tanh`). If you pass `None`, no activation
is applied (ie. "linear" activation: `a(x) = x`).
recurrent_activation: Activation function to use for the recurrent step.
Default: sigmoid (`sigmoid`). If you pass `None`, no activation is
applied (ie. "linear" activation: `a(x) = x`).
use_bias: Boolean (default `True`), whether the layer uses a bias vector.
kernel_initializer: Initializer for the `kernel` weights matrix, used for
the linear transformation of the inputs. Default: `glorot_uniform`.
recurrent_initializer: Initializer for the `recurrent_kernel` weights
matrix, used for the linear transformation of the recurrent state.
Default: `orthogonal`.
bias_initializer: Initializer for the bias vector. Default: `zeros`.
unit_forget_bias: Boolean (default `True`). If True, add 1 to the bias of
the forget gate at initialization. Setting it to true will also force
`bias_initializer="zeros"`. This is recommended in [Jozefowicz et
          al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
kernel_regularizer: Regularizer function applied to the `kernel` weights
matrix. Default: `None`.
recurrent_regularizer: Regularizer function applied to the
`recurrent_kernel` weights matrix. Default: `None`.
bias_regularizer: Regularizer function applied to the bias vector. Default:
`None`.
activity_regularizer: Regularizer function applied to the output of the
layer (its "activation"). Default: `None`.
kernel_constraint: Constraint function applied to the `kernel` weights
matrix. Default: `None`.
recurrent_constraint: Constraint function applied to the `recurrent_kernel`
weights matrix. Default: `None`.
bias_constraint: Constraint function applied to the bias vector. Default:
`None`.
dropout: Float between 0 and 1. Fraction of the units to drop for the linear
transformation of the inputs. Default: 0.
recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for
the linear transformation of the recurrent state. Default: 0.
return_sequences: Boolean. Whether to return the last output. in the output
sequence, or the full sequence. Default: `False`.
return_state: Boolean. Whether to return the last state in addition to the
output. Default: `False`.
go_backwards: Boolean (default `False`). If True, process the input sequence
backwards and return the reversed sequence.
stateful: Boolean (default `False`). If True, the last state for each sample
at index i in a batch will be used as initial state for the sample of
index i in the following batch.
time_major: The shape format of the `inputs` and `outputs` tensors.
If True, the inputs and outputs will be in shape
`[timesteps, batch, feature]`, whereas in the False case, it will be
`[batch, timesteps, feature]`. Using `time_major = True` is a bit more
efficient because it avoids transposes at the beginning and end of the
RNN calculation. However, most TensorFlow data is batch-major, so by
default this function accepts input and emits output in batch-major
form.
unroll: Boolean (default `False`). If True, the network will be unrolled,
else a symbolic loop will be used. Unrolling can speed-up a RNN, although
it tends to be more memory-intensive. Unrolling is only suitable for short
sequences.
 */
public class LSTM(
    public val units: Int = 128,
    public val activation: Activations = Activations.Tanh,
    public val recurrentActivation: Activations = Activations.Sigmoid,
    public val kernelInitializer: Initializer = GlorotUniform(),
    public val biasInitializer: Initializer = GlorotUniform(),
    public val useBias: Boolean = true,
    public val unitForgetBias: Boolean = true,
    public val dropout: Float = 0.0f,
    public val recurrentDropout: Float = 0.0f,
    public val returnSequences: Boolean = true,
    public val returnState: Boolean = true,
    public val goBackwards: Boolean = true,
    public val stateful: Boolean = true,
    public val timeMajor: Boolean = true,
    public val unroll: Boolean = true,
    name: String = "",
) : Layer(name) {

    private lateinit var weightShape: Shape
    private lateinit var gamma: Variable<Float>
    private lateinit var beta: Variable<Float>
    private lateinit var movingMean: Variable<Float>
    private lateinit var movingVariance: Variable<Float>

    override fun defineVariables(tf: Ops, kGraph: KGraph, inputShape: Shape) {
        // Compute shapes of kernel and bias matrices
        weightShape = Shape.make(inputShape.size(inputShape.numDimensions() - 1))

        /*if (name.isNotEmpty()) {
            val gammaVariableName = batchNormGammaVarName(name)
            val betaVariableName = batchNormBetaVarName(name)
            val movingMeanVariableName = batchNormMovingMeanVarName(name)
            val movingVarianceVariableName = batchNormMovingVarianceVarName(name)

            gamma = tf.withName(gammaVariableName).variable(weightShape, getDType())
            beta = tf.withName(betaVariableName).variable(weightShape, getDType())
            movingMean = tf.withName(movingMeanVariableName).variable(weightShape, getDType())
            movingVariance = tf.withName(movingVarianceVariableName).variable(weightShape, getDType())

            isTrainable = false // TODO: add isTrainable to addWeight method as a flag
            gamma = addWeight(tf, kGraph, gammaVariableName, gamma, gammaInitializer)
            beta = addWeight(tf, kGraph, betaVariableName, beta, betaInitializer)
            movingMean = addWeight(tf, kGraph, movingMeanVariableName, movingMean, movingMeanInitializer)
            movingVariance =
                addWeight(tf, kGraph, movingVarianceVariableName, movingVariance, movingVarianceInitializer)
        }*/
    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        return inputShape
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Boolean,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        return input
    }


    override fun getWeights(): List<Array<*>> {
        return emptyList()
    }

    override fun hasActivation(): Boolean {
        return false
    }

    override fun getParams(): Int {
        return 0
    }

    fun getWeightShape(): LongArray {
        return TensorShape(weightShape).dims()
    }
}
