/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.initializer.Constant
import org.jetbrains.kotlinx.dl.api.core.initializer.Zeros
import org.jetbrains.kotlinx.dl.api.core.layer.Conv2DTest.Companion.create2DTensor
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2DTranspose
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.junit.jupiter.api.Test

class Conv2DTransposeTest : ConvLayerTest() {
    @Test
    fun testZeroInput() {
        val input = create2DTensor(batchSize = 1, height = 2, width = 3, channels = 32, initValue = 0.0f)
        val expected = create2DTensor(batchSize = 1, height = 2, width = 3, channels = 3, initValue = 0.0f)

        assertTensorsEquals(
            Conv2DTranspose(
                3, 3,
                name = "TestConv2DTranspose_zeroInput",
                biasInitializer = Zeros()
            ),
            input,
            expected
        )
    }

    @Test
    fun noPadding() {
        val input = create2DTensor(batchSize = 1, height = 2, width = 3, channels = 32, initValue = 1f)
        val expected = create2DTensor(
            batchSize = 1, channels = 3,
            floatArrayOf(32f, 64f, 96f, 64f, 32f),
            floatArrayOf(64f, 128f, 192f, 128f, 64f),
            floatArrayOf(64f, 128f, 192f, 128f, 64f),
            floatArrayOf(32f, 64f, 96f, 64f, 32f)
        )

        assertTensorsEquals(
            Conv2DTranspose(
                kernelSize = 3,
                name = "TestConv2DTranspose_noPadding",
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
        val input = create2DTensor(batchSize = 1, height = 2, width = 3, channels = 32, initValue = 1f)
        val expected = create2DTensor(
            batchSize = 1, channels = 3,
            floatArrayOf(128f, 192f, 128f),
            floatArrayOf(128f, 192f, 128f)
        )

        assertTensorsEquals(
            Conv2DTranspose(
                kernelSize = 3,
                name = "TestConv2DTranspose_samePadding",
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
        val input = create2DTensor(batchSize = 1, height = 2, width = 3, channels = 32, initValue = 1f)
        val expected = create2DTensor(
            batchSize = 1, channels = 3,
            floatArrayOf(32f, 64f, 96f, 64f, 32f),
            floatArrayOf(64f, 128f, 192f, 128f, 64f),
            floatArrayOf(64f, 128f, 192f, 128f, 64f),
            floatArrayOf(32f, 64f, 96f, 64f, 32f)
        )

        assertTensorsEquals(
            Conv2DTranspose(
                kernelSize = 3,
                name = "TestConv2DTranspose_outputPadding",
                filters = 3,
                kernelInitializer = Constant(1.0f),
                biasInitializer = Zeros(),
                padding = ConvPadding.SAME,
                outputPadding = intArrayOf(0, 0, 1, 1, 1, 1, 0, 0)
            ),
            input,
            expected
        )
    }

    @Test
    fun noPaddingStrides() {
        val input = create2DTensor(batchSize = 1, height = 2, width = 3, channels = 32, initValue = 1f)
        val expected = create2DTensor(
            batchSize = 1, channels = 3,
            floatArrayOf(32f, 32f, 64f, 32f, 64f, 32f, 32f),
            floatArrayOf(32f, 32f, 64f, 32f, 64f, 32f, 32f),
            floatArrayOf(64f, 64f, 128f, 64f, 128f, 64f, 64f),
            floatArrayOf(32f, 32f, 64f, 32f, 64f, 32f, 32f),
            floatArrayOf(32f, 32f, 64f, 32f, 64f, 32f, 32f)
        )

        assertTensorsEquals(
            Conv2DTranspose(
                name = "TestConv2DTranspose_noPadding_strides",
                filters = 3,
                kernelInitializer = Constant(1.0f),
                biasInitializer = Zeros(),
                padding = ConvPadding.VALID,
                strides = intArrayOf(1, 2, 2, 1)
            ),
            input,
            expected
        )
    }

    @Test
    fun samePaddingStrides() {
        val input = create2DTensor(batchSize = 1, height = 2, width = 3, channels = 32, initValue = 1f)
        val expected = create2DTensor(
            batchSize = 1, channels = 3,
            floatArrayOf(32f, 32f, 64f, 32f, 64f, 32f),
            floatArrayOf(32f, 32f, 64f, 32f, 64f, 32f),
            floatArrayOf(64f, 64f, 128f, 64f, 128f, 64f),
            floatArrayOf(32f, 32f, 64f, 32f, 64f, 32f)
        )

        assertTensorsEquals(
            Conv2DTranspose(
                name = "TestConv2DTranspose_samePadding_strides",
                filters = 3,
                kernelInitializer = Constant(1.0f),
                biasInitializer = Zeros(),
                padding = ConvPadding.SAME,
                strides = intArrayOf(1, 2, 2, 1)
            ),
            input,
            expected
        )
    }

    @Test
    fun outputPaddingStrides() {
        val input = create2DTensor(batchSize = 1, height = 2, width = 3, channels = 32, initValue = 1f)
        val expected = create2DTensor(
            batchSize = 1, channels = 3,
            floatArrayOf(32f, 32f, 64f, 32f, 64f, 32f, 32f),
            floatArrayOf(32f, 32f, 64f, 32f, 64f, 32f, 32f),
            floatArrayOf(64f, 64f, 128f, 64f, 128f, 64f, 64f),
            floatArrayOf(32f, 32f, 64f, 32f, 64f, 32f, 32f),
            floatArrayOf(32f, 32f, 64f, 32f, 64f, 32f, 32f)
        )

        assertTensorsEquals(
            Conv2DTranspose(
                name = "TestConv2DTranspose_outputPadding_strides",
                filters = 3,
                kernelInitializer = Constant(1.0f),
                biasInitializer = Zeros(),
                padding = ConvPadding.SAME,
                outputPadding = intArrayOf(0, 0, 1, 1, 1, 1, 0, 0),
                strides = intArrayOf(1, 2, 2, 1)
            ),
            input,
            expected
        )
    }
}