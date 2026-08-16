/*
 * Copyright 2021-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.activation

import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.types.TBool
import org.tensorflow.types.TFloat32

/**
 * Base class for all layer class representing activation functions.
 *
 * By default, it is marked as __not trainable__ layer with no extra
 * parameters and weights but having the activation on it.
 *
 * By default, it defines returning the output with the same shape
 * as the input Operand.
 *
 * @param [name] Layer name. Would be changed if empty during model compilation.
 */
public abstract class AbstractActivationLayer(name: String) : Layer(name) {
    /**
     * Applies the activation functions to the [input] to produce the output.
     *
     * @param [tf] TensorFlow graph API for building operations.
     * @param [input] TensorFlow graph leaf node representing layer output before activation function.
     */
    public abstract fun forward(
        tf: Ops,
        input: Operand<TFloat32>
    ): Operand<TFloat32>

    override fun build(
        tf: Ops,
        input: Operand<TFloat32>,
        isTraining: Operand<TBool>,
        numberOfLosses: Operand<TFloat32>?
    ): Operand<TFloat32> = forward(tf, input)

    override val hasActivation: Boolean get() = true
}
