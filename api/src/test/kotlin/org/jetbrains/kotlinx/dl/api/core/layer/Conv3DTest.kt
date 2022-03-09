/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.initializer.Constant
import org.jetbrains.kotlinx.dl.api.core.initializer.Zeros
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv3D
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.junit.jupiter.api.Test

internal class Conv3DTest : ConvLayerTest() {
    @Test
    fun zeroedInputTensorWithDefaultValues() {
        val input = create3DTensor(batchSize = 1, depth = 3, height = 3, width = 3, channels = 1, initValue = 0.0f)
        val expected = create3DTensor(batchSize = 1, depth = 3, height = 3, width = 3, channels = 32, initValue = 0.0f)

        assertTensorsEquals(
            Conv3D(
                32, 3,
                name = "TestConv3D_1",
                biasInitializer = Zeros()
            ),
            input,
            expected
        )
    }

    @Test
    fun constantInputTensorWithValidPadding() {
        val input = create3DTensor(batchSize = 1, depth = 3, height = 3, width = 3, channels = 1, initValue = 1.0f)
        val expected = create3DTensor(batchSize = 1, depth = 2, height = 2, width = 2, channels = 16, initValue = 8.0f)

        assertTensorsEquals(
            Conv3D(
                name = "TestConv3D_2",
                filters = 16,
                kernelInitializer = Constant(1.0f),
                biasInitializer = Zeros(),
                kernelSize = intArrayOf(2, 2, 2),
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
                    arrayOf(
                        floatArrayOf(0.8373f, 0.8765f, 0.4692f),
                        floatArrayOf(0.5244f, 0.6573f, 0.9453f)
                    ),
                    arrayOf(
                        floatArrayOf(0.6919f, 0.0724f, 0.7274f),
                        floatArrayOf(0.1452f, 0.9262f, 0.7690f)
                    )
                ),
                arrayOf(
                    arrayOf(
                        floatArrayOf(0.453f, 0.3465f, 0.4342f),
                        floatArrayOf(0.2344f, 0.9673f, 0.1953f)
                    ),
                    arrayOf(
                        floatArrayOf(0.9222f, 0.8924f, 0.7234f),
                        floatArrayOf(0.2345f, 0.2622f, 0.9012f)
                    )
                )
            )
        )
        val expected = arrayOf(arrayOf(arrayOf(arrayOf(floatArrayOf(input.sum())))))

        assertTensorsEquals(
            Conv3D(
                name = "TestConv3D_3",
                filters = 1,
                kernelInitializer = Constant(1.0f),
                biasInitializer = Zeros(),
                kernelSize = intArrayOf(2, 2, 2),
                padding = ConvPadding.VALID
            ),
            input,
            expected
        )
    }

    internal companion object {
        internal fun create3DTensor(
            batchSize: Int,
            depth: Int,
            height: Int,
            width: Int,
            channels: Int,
            initValue: Float
        ) = Array(batchSize) { Array(depth) { Array(height) { Array(width) { FloatArray(channels) { initValue } } } } }

        internal fun create3DTensor(
            batchSize: Int,
            channels: Int,
            vararg frames: Array<FloatArray>
        ) = Array(batchSize) {
            Array(frames.size) { frameIdx ->
                val rows = frames[frameIdx]
                Array(rows.size) { rowIdx ->
                    Array(rows[rowIdx].size) { columnIdx -> FloatArray(channels) { rows[rowIdx][columnIdx] } }
                }
            }
        }
    }
}
