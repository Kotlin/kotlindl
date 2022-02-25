/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.initializer.Constant
import org.jetbrains.kotlinx.dl.api.core.initializer.Zeros
import org.jetbrains.kotlinx.dl.api.core.layer.Conv3DTest.Companion.create3DTensor
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv3DTranspose
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.junit.jupiter.api.Test

class Conv3DTransposeTest : ConvLayerTest() {
    @Test
    fun testZeroInput() {
        val input = create3DTensor(batchSize = 1, depth = 1, height = 2, width = 3, channels = 32, initValue = 0.0f)
        val expected = create3DTensor(batchSize = 1, depth = 1, height = 2, width = 3, channels = 3, initValue = 0.0f)

        assertTensorsEquals(
            Conv3DTranspose(
                3, 3,
                name = "TestConv3DTranspose_zeroInput",
                biasInitializer = Zeros()
            ),
            input,
            expected
        )
    }

    @Test
    fun noPadding() {
        val input = create3DTensor(batchSize = 1, depth = 1, height = 2, width = 3, channels = 32, initValue = 1f)
        val expected = create3DTensor(
            batchSize = 1, channels = 3,
            arrayOf(
                floatArrayOf(32f, 64f, 96f, 64f, 32f),
                floatArrayOf(64f, 128f, 192f, 128f, 64f),
                floatArrayOf(64f, 128f, 192f, 128f, 64f),
                floatArrayOf(32f, 64f, 96f, 64f, 32f)
            ),
            arrayOf(
                floatArrayOf(32f, 64f, 96f, 64f, 32f),
                floatArrayOf(64f, 128f, 192f, 128f, 64f),
                floatArrayOf(64f, 128f, 192f, 128f, 64f),
                floatArrayOf(32f, 64f, 96f, 64f, 32f)
            ),
            arrayOf(
                floatArrayOf(32f, 64f, 96f, 64f, 32f),
                floatArrayOf(64f, 128f, 192f, 128f, 64f),
                floatArrayOf(64f, 128f, 192f, 128f, 64f),
                floatArrayOf(32f, 64f, 96f, 64f, 32f)
            )
        )

        assertTensorsEquals(
            Conv3DTranspose(
                kernelSize = 3,
                name = "TestConv3DTranspose_noPadding",
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
        val input = create3DTensor(batchSize = 1, depth = 1, height = 2, width = 3, channels = 32, initValue = 1f)
        val expected = create3DTensor(
            batchSize = 1, channels = 3,
            arrayOf(
                floatArrayOf(128f, 192f, 128f),
                floatArrayOf(128f, 192f, 128f)
            )
        )

        assertTensorsEquals(
            Conv3DTranspose(
                kernelSize = 3,
                name = "TestConv3DTranspose_samePadding",
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
    fun noPaddingStrides() {
        val input = create3DTensor(batchSize = 1, depth = 1, height = 2, width = 3, channels = 32, initValue = 1f)
        val expected = create3DTensor(
            batchSize = 1, channels = 3,
            arrayOf(
                floatArrayOf(32f, 32f, 64f, 32f, 64f, 32f, 32f),
                floatArrayOf(32f, 32f, 64f, 32f, 64f, 32f, 32f),
                floatArrayOf(64f, 64f, 128f, 64f, 128f, 64f, 64f),
                floatArrayOf(32f, 32f, 64f, 32f, 64f, 32f, 32f),
                floatArrayOf(32f, 32f, 64f, 32f, 64f, 32f, 32f)
            ),
            arrayOf(
                floatArrayOf(32f, 32f, 64f, 32f, 64f, 32f, 32f),
                floatArrayOf(32f, 32f, 64f, 32f, 64f, 32f, 32f),
                floatArrayOf(64f, 64f, 128f, 64f, 128f, 64f, 64f),
                floatArrayOf(32f, 32f, 64f, 32f, 64f, 32f, 32f),
                floatArrayOf(32f, 32f, 64f, 32f, 64f, 32f, 32f)
            ),
            arrayOf(
                floatArrayOf(32f, 32f, 64f, 32f, 64f, 32f, 32f),
                floatArrayOf(32f, 32f, 64f, 32f, 64f, 32f, 32f),
                floatArrayOf(64f, 64f, 128f, 64f, 128f, 64f, 64f),
                floatArrayOf(32f, 32f, 64f, 32f, 64f, 32f, 32f),
                floatArrayOf(32f, 32f, 64f, 32f, 64f, 32f, 32f)
            )
        )

        assertTensorsEquals(
            Conv3DTranspose(
                name = "TestConv3DTranspose_noPadding_strides",
                filters = 3,
                kernelInitializer = Constant(1.0f),
                biasInitializer = Zeros(),
                padding = ConvPadding.VALID,
                strides = intArrayOf(1, 2, 2, 2, 1)
            ),
            input,
            expected
        )
    }

    @Test
    fun samePaddingStrides() {
        val input = create3DTensor(batchSize = 1, depth = 1, height = 2, width = 3, channels = 32, initValue = 1f)
        val expected = create3DTensor(
            batchSize = 1, channels = 3,
            arrayOf(
                floatArrayOf(32f, 32f, 64f, 32f, 64f, 32f),
                floatArrayOf(32f, 32f, 64f, 32f, 64f, 32f),
                floatArrayOf(64f, 64f, 128f, 64f, 128f, 64f),
                floatArrayOf(32f, 32f, 64f, 32f, 64f, 32f),
            ),
            arrayOf(
                floatArrayOf(32f, 32f, 64f, 32f, 64f, 32f),
                floatArrayOf(32f, 32f, 64f, 32f, 64f, 32f),
                floatArrayOf(64f, 64f, 128f, 64f, 128f, 64f),
                floatArrayOf(32f, 32f, 64f, 32f, 64f, 32f),
            )
        )

        assertTensorsEquals(
            Conv3DTranspose(
                name = "TestConv3DTranspose_samePadding_strides",
                filters = 3,
                kernelInitializer = Constant(1.0f),
                biasInitializer = Zeros(),
                padding = ConvPadding.SAME,
                strides = intArrayOf(1, 2, 2, 2, 1)
            ),
            input,
            expected
        )
    }
}