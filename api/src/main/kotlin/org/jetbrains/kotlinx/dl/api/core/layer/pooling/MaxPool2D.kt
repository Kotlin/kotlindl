/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.pooling

import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.shape.convOutputLength
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

/**
 * Max pooling layer for 2D inputs (e.g. images).
 *
 * NOTE: Works with tensors which must have rank 4 (batch, height, width, channels).
 *
 * @property [poolSize] The size of the sliding window for each dimension of input tensor (pool batch, pool height, pool width, pool channels).
 * Usually, pool batch and pool channels are equal to 1.
 * @property [strides] Strides of the pooling operation for each dimension of input tensor.
 * @property [padding] The padding method, either 'valid' or 'same' or 'full'.
 * @property [name] Custom layer name.
 * @constructor Creates [MaxPool2D] object.
 */
public class MaxPool2D(
    public val poolSize: IntArray = intArrayOf(1, 2, 2, 1),
    public val strides: IntArray = intArrayOf(1, 2, 2, 1),
    public val padding: ConvPadding = ConvPadding.VALID,
    name: String = ""
) : Layer(name) {
    public constructor(
        poolSize: Int = 2,
        strides: Int = 2,
        padding: ConvPadding = ConvPadding.VALID,
        name: String = ""
    ) : this(
        poolSize = intArrayOf(1, poolSize, poolSize, 1),
        strides = intArrayOf(1, strides, strides, 1),
        padding = padding,
        name = name
    )

    override fun build(tf: Ops, inputShape: Shape) {}

    override fun computeOutputShape(inputShape: Shape): Shape {
        var rows = inputShape.size(1)
        var cols = inputShape.size(2)
        rows = convOutputLength(
            rows, poolSize[1], padding,
            strides[1]
        )
        cols = convOutputLength(
            cols, poolSize[2], padding,
            strides[2]
        )

        return Shape.make(inputShape.size(0), rows, cols, inputShape.size(3))
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        val paddingName = padding.paddingName

        return tf.nn.maxPool(
            input,
            tf.constant(poolSize),
            tf.constant(strides),
            paddingName
        )
    }

    override fun toString(): String {
        return "MaxPool2D(name = $name, poolSize=${poolSize.contentToString()}, strides=${strides.contentToString()}, padding=$padding, hasActivation=$hasActivation)"
    }

    override var weights: Map<String, Array<*>>
        get() = emptyMap()
        set(value) = assignWeights(value)

    override val hasActivation: Boolean get() = false
}
