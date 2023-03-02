/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.activation

import org.tensorflow.Operand
import org.tensorflow.op.Ops

/**
 * Basic interface for all activation functions.
 */
public interface Activation {
    /**
     * Applies the activation functions to the input [features] to produce the output.
     *
     * @param [tf] TensorFlow graph API for building operations.
     * @param [features] TensorFlow graph leaf node representing layer output before activation function.
     * @param [name] Activation name for TensorFlow graph building purposes.
     */
    public fun apply(tf: Ops, features: Operand<Float>, name: String = ""): Operand<Float> {
        return if (name.isEmpty()) features else tf.withName("Activation_$name").identity(apply(tf, features))
    }

    /**
     * Applies the activation functions to the input [features] to produce the output.
     *
     * @param [tf] TensorFlow graph API for building operations.
     * @param [features] TensorFlow graph leaf node representing layer output before activation function.
     */
    public fun apply(tf: Ops, features: Operand<Float>): Operand<Float>
}