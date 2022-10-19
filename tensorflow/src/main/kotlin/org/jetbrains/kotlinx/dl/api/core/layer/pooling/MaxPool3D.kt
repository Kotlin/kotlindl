/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.pooling

import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.tensorflow.Operand
import org.tensorflow.op.Ops
import java.util.*

/**
 * Max pooling operation for 3D data (spatial or spatio-temporal).
 * NOTE: Works with tensors which must have rank 5 (batch, depth, height, width, channels).
 * @property [poolSize] The size of the sliding window for each dimension of input tensor (pool batch, pool depth ,pool height, pool width, pool channels).
 * Usually, pool batch and pool channels are equal to 1.
 * @property [strides] Strides of the pooling operation for each dimension of input tensor.
 * @property [padding] The padding method, either 'valid' or 'same'.
 * @property [name] Custom layer name.
 * @constructor Creates [MaxPool2D] object.
 *
 * @since 0.3
 */
public class MaxPool3D(
    public var poolSize: IntArray = intArrayOf(1, 2, 2, 2, 1),
    public var strides: IntArray = intArrayOf(1, 2, 2, 2, 1),
    public val padding: ConvPadding = ConvPadding.VALID,
    name: String = ""
) : Layer(name) {
    public constructor(
        poolSize: Int = 2,
        strides: Int = 2,
        padding: ConvPadding = ConvPadding.VALID,
        name: String = ""
    ) : this(
        poolSize = intArrayOf(1, poolSize, poolSize, poolSize, 1),
        strides = intArrayOf(1, strides, strides, strides, 1),
        padding = padding,
        name = name
    )

    override fun build(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        val paddingName = padding.paddingName
        val tfPoolSize = Arrays.stream(poolSize).asLongStream().toArray()
        val tfStrides = Arrays.stream(strides).asLongStream().toArray()
        val tfInput: Operand<Float> = input

        return tf.nn.maxPool3d(tfInput, tfPoolSize.toList(), tfStrides.toList(), paddingName)
    }

    override fun toString(): String {
        return "MaxPool3D(name = $name, poolSize=${poolSize.contentToString()}, strides=${strides.contentToString()}, padding=$padding, hasActivation=$hasActivation)"
    }

    override val hasActivation: Boolean get() = false
}
