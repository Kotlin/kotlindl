/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.initializer.Constant
import org.jetbrains.kotlinx.dl.api.core.initializer.Zeros
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv1D
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.junit.jupiter.api.Test

internal class Conv1DTest : ConvLayerTest() {
    @Test
    fun zeroedInputTensorWithDefaultValues() {
        val input = create1DTensor(batchSize = 1, size = 3, channels = 1, initValue = 0.0f)
        val expected = create1DTensor(batchSize = 1, size = 3, channels = 32, initValue = 0.0f)

        assertTensorsEquals(
            Conv1D(
                32,
                3,
                1,
                1,
                name = "TestConv1D_1",
                biasInitializer = Zeros()
            ),
            input,
            expected
        )
    }

    @Test
    fun constantInputTensorWithValidPadding() {
        val input = create1DTensor(batchSize = 1, size = 3, channels = 1, initValue = 1.0f)
        val expected = create1DTensor(batchSize = 1, size = 2, channels = 16, initValue = 2.0f)

        assertTensorsEquals(
            Conv1D(
                strides = 1,
                name = "TestConv1D_2",
                filters = 16,
                kernelInitializer = Constant(1.0f),
                biasInitializer = Zeros(),
                kernelLength = 2,
                padding = ConvPadding.VALID
            ),
            input,
            expected
        )
    }

    @Test
    fun randomInputTensorWithOnesWeight() {
        val input = arrayOf(
            arrayOf(
                floatArrayOf(0.5967f, 0.6496f, 0.1336f, 0.0338f),
                floatArrayOf(0.7829f, 0.2899f, 0.2759f, 0.0719f),
                floatArrayOf(0.0820f, 0.2821f, 0.7951f, 0.3663f)
            )
        )
        val expected = arrayOf(arrayOf(floatArrayOf(input.sum())))

        assertTensorsEquals(
            Conv1D(
                strides = 1,
                name = "TestConv1D_3",
                filters = 1,
                kernelInitializer = Constant(1.0f),
                biasInitializer = Zeros(),
                kernelLength = 3,
                padding = ConvPadding.VALID
            ),
            input,
            expected
        )
    }

    companion object {
        internal fun create1DTensor(
            batchSize: Int,
            size: Int,
            channels: Int,
            initValue: Float
        ) = Array(batchSize) { Array(size) { FloatArray(channels) { initValue } } }

        internal fun create1DTensor(
            batchSize: Int,
            channels: Int,
            sequence: FloatArray,
        ) = Array(batchSize) { Array(sequence.size) { idx -> FloatArray(channels) { sequence[idx] } } }
    }
}
