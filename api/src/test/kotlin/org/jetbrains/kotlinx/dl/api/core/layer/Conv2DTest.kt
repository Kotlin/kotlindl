/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.initializer.Constant
import org.jetbrains.kotlinx.dl.api.core.initializer.Zeros
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.junit.jupiter.api.Test

internal class Conv2DTest : ConvLayerTest() {
    @Test
    fun zeroedInputTensorWithDefaultValues() {
        val input = create2DTensor(batchSize = 1, height = 3, width = 3, channels = 1, initValue = 0.0f)
        val expected = create2DTensor(batchSize = 1, height = 3, width = 3, channels = 32, initValue = 0.0f)

        assertTensorsEquals(
            Conv2D(
                32,
                3,
                name = "TestConv2D_1",
                biasInitializer = Zeros()
            ),
            input,
            expected
        )
    }

    @Test
    fun constantInputTensorWithValidPadding() {
        val input = create2DTensor(batchSize = 1, height = 3, width = 3, channels = 1, initValue = 1.0f)
        val expected = create2DTensor(batchSize = 1, height = 2, width = 2, channels = 16, initValue = 4.0f)

        assertTensorsEquals(
            Conv2D(
                name = "TestConv2D_2",
                filters = 16,
                kernelInitializer = Constant(1.0f),
                biasInitializer = Zeros(),
                kernelSize = intArrayOf(2, 2),
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
                arrayOf(
                    floatArrayOf(0.8373f, 0.8765f, 0.4692f),
                    floatArrayOf(0.5244f, 0.6573f, 0.9453f)
                ),

                arrayOf(
                    floatArrayOf(0.6919f, 0.0724f, 0.7274f),
                    floatArrayOf(0.1452f, 0.9262f, 0.7690f)
                )
            )
        )
        val expected = arrayOf(arrayOf(arrayOf(floatArrayOf(input.sum()))))

        assertTensorsEquals(
            Conv2D(
                name = "TestConv2D_3",
                filters = 1,
                kernelInitializer = Constant(1.0f),
                biasInitializer = Zeros(),
                kernelSize = intArrayOf(2, 2),
                padding = ConvPadding.VALID
            ),
            input,
            expected
        )
    }

    internal companion object {
        internal fun create2DTensor(
            batchSize: Int,
            height: Int,
            width: Int,
            channels: Int,
            initValue: Float
        ) = Array(batchSize) { Array(height) { Array(width) { FloatArray(channels) { initValue } } } }

        internal fun create2DTensor(
            batchSize: Int,
            channels: Int,
            vararg rows: FloatArray
        ) = Array(batchSize) {
            Array(rows.size) { rowIdx ->
                Array(rows[rowIdx].size) { columnIdx -> FloatArray(channels) { rows[rowIdx][columnIdx] } }
            }
        }
    }
}
