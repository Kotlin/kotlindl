/*
 * Copyright 2021-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.initializer.Constant
import org.jetbrains.kotlinx.dl.api.core.initializer.Zeros
import org.jetbrains.kotlinx.dl.api.core.layer.Conv1DTest.Companion.create1DTensor
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv1DTranspose
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.junit.jupiter.api.Test

class Conv1DTransposeTest : ConvLayerTest() {
    @Test
    fun zeroInput() {
        val input = create1DTensor(batchSize = 1, size = 3, channels = 32, initValue = 0f)
        val expected = create1DTensor(batchSize = 1, size = 3, channels = 3, initValue = 0f)

        assertTensorsEquals(
            Conv1DTranspose(
                name = "TestConv1DTranspose_zeroInput",
                filters = 3,
                biasInitializer = Zeros()
            ),
            input,
            expected
        )
    }

    @Test
    fun noPadding() {
        val input = create1DTensor(batchSize = 1, size = 3, channels = 32, initValue = 1f)
        val expected = create1DTensor(batchSize = 1, channels = 3, floatArrayOf(32f, 64f, 96f, 64f, 32f))

        assertTensorsEquals(
            Conv1DTranspose(
                name = "TestConv1DTranspose_noPadding",
                filters = 3,
                kernelInitializer = Constant(1.0f),
                biasInitializer = Zeros(),
                padding = ConvPadding.VALID
            ),
            input,
            expected
        )
    }

    @Test
    fun samePadding() {
        val input = create1DTensor(batchSize = 1, size = 3, channels = 32, initValue = 1f)
        val expected = create1DTensor(batchSize = 1, channels = 3, floatArrayOf(64f, 96f, 64f))

        assertTensorsEquals(
            Conv1DTranspose(
                name = "TestConv1DTranspose_samePadding",
                filters = 3,
                kernelInitializer = Constant(1.0f),
                biasInitializer = Zeros(),
                padding = ConvPadding.SAME
            ),
            input,
            expected
        )
    }

    @Test
    fun outputPadding() {
        val input = create1DTensor(batchSize = 1, size = 3, channels = 32, initValue = 1f)
        val expected = create1DTensor(batchSize = 1, channels = 3, floatArrayOf(32f, 64f, 96f, 64f, 32f))

        assertTensorsEquals(
            Conv1DTranspose(
                name = "TestConv1DTranspose_outputPadding",
                filters = 3,
                kernelInitializer = Constant(1.0f),
                biasInitializer = Zeros(),
                padding = ConvPadding.SAME,
                outputPadding = intArrayOf(0, 0, 1, 1, 0, 0)
            ),
            input,
            expected
        )
    }

    @Test
    fun noPaddingStrides() {
        val input = create1DTensor(batchSize = 1, size = 3, channels = 32, initValue = 1f)
        val expected = create1DTensor(batchSize = 1, channels = 3, floatArrayOf(32f, 32f, 64f, 32f, 64f, 32f, 32f))

        assertTensorsEquals(
            Conv1DTranspose(
                name = "TestConv1DTranspose_noPadding_strides",
                filters = 3,
                kernelInitializer = Constant(1.0f),
                biasInitializer = Zeros(),
                padding = ConvPadding.VALID,
                strides = intArrayOf(1, 2, 1)
            ),
            input,
            expected
        )
    }

    @Test
    fun samePaddingStrides() {
        val input = create1DTensor(batchSize = 1, size = 3, channels = 32, initValue = 1f)
        val expected = create1DTensor(batchSize = 1, channels = 3, floatArrayOf(32f, 32f, 64f, 32f, 64f, 32f))

        assertTensorsEquals(
            Conv1DTranspose(
                name = "TestConv1DTranspose_samePadding_strides",
                filters = 3,
                kernelInitializer = Constant(1.0f),
                biasInitializer = Zeros(),
                padding = ConvPadding.SAME,
                strides = intArrayOf(1, 2, 1)
            ),
            input,
            expected
        )
    }

    @Test
    fun outputPaddingStrides() {
        val input = create1DTensor(batchSize = 1, size = 3, channels = 32, initValue = 1f)
        val expected = create1DTensor(batchSize = 1, channels = 3, floatArrayOf(32f, 32f, 64f, 32f, 64f, 32f, 32f))

        assertTensorsEquals(
            Conv1DTranspose(
                name = "TestConv1DTranspose_outputPadding_strides",
                filters = 3,
                kernelInitializer = Constant(1.0f),
                biasInitializer = Zeros(),
                padding = ConvPadding.SAME,
                outputPadding = intArrayOf(0, 0, 1, 1, 0, 0),
                strides = intArrayOf(1, 2, 1)
            ),
            input,
            expected
        )
    }
}