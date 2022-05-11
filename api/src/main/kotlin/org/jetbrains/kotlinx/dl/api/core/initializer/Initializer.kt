/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.initializer

import org.jetbrains.kotlinx.dl.api.core.layer.InitializerOperation
import org.jetbrains.kotlinx.dl.api.core.shape.shapeOperand
import org.jetbrains.kotlinx.dl.api.core.util.defaultAssignOpName
import org.jetbrains.kotlinx.dl.api.core.util.defaultInitializerOpName
import org.tensorflow.Operand
import org.tensorflow.op.Ops

/**
 * Initializer base class: all initializers inherit this class.
 *
 * Initializers allow you to pre-specify an initialization strategy, encoded in
 * the Initializer object, without knowing the shape and dtype of the variable
 * being initialized.
 */
public abstract class Initializer {
    /**
     * Adds an `Assign` Op to the graph to initialize
     * a tensorflow variable as specified by the initializer.
     *
     * @param [fanIn] The maximum number of inputs that an initializer can accept.
     * @param [fanOut] The maximum number of inputs that the output of an initializer can feed to other steps.
     * @param [tf] Tensorflow Ops Accessor
     * @param [input] Variable to initialize
     * @return initializer operation.
     * @see InitializerOperation
     */
    public fun apply(
        fanIn: Int,
        fanOut: Int,
        tf: Ops,
        input: Operand<Float>,
        name: String
    ): InitializerOperation {
        val initialize = initialize(
            fanIn, fanOut, tf,
            shapeOperand(tf, input.asOutput().shape()), defaultInitializerOpName(name)
        )
        return InitializerOperation(tf.withName(defaultAssignOpName(name)).assign(input, initialize), initialize)
    }


    /**
     * Returns a Tensor object initialized as specified by the initializer.
     *
     * @param [fanIn] The maximum number of inputs that an initializer can accept.
     * @param [fanOut] The maximum number of inputs that the output of an initializer can feed to other steps.
     * @param [tf] Tensorflow Ops Accessor.
     * @param [shape] Shape of the tensor.
     * @param [name] Initializer name.
     */
    public abstract fun initialize(
        fanIn: Int,
        fanOut: Int,
        tf: Ops,
        shape: Operand<Int>,
        name: String
    ): Operand<Float>
}