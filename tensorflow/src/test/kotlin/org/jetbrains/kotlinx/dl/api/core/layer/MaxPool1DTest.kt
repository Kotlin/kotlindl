/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.MaxPool1D
import org.jetbrains.kotlinx.dl.api.core.shape.toIntArray
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test
import org.tensorflow.EagerSession
import org.tensorflow.op.Ops

internal class MaxPool1DTest {
    @Test
    fun default() {
        val input = arrayOf(
            arrayOf(
                floatArrayOf(1.0f, -2.0f, 3.0f),
                floatArrayOf(0.5f, 2.0f, 5.0f),
                floatArrayOf(-1.0f, 3.0f, 2.0f),
                floatArrayOf(1.5f, -1.0f, 0.5f)
            ),
            arrayOf(
                floatArrayOf(5.0f, 3.0f, 1.0f),
                floatArrayOf(6.0f, -2.5f, 4.0f),
                floatArrayOf(7.0f, 0.0f, 5.0f),
                floatArrayOf(1.0f, 2.0f, 4.0f)
            ),
        )
        val expected = arrayOf(
            arrayOf(
                floatArrayOf(1.0f, 2.0f, 5.0f),
                floatArrayOf(1.5f, 3.0f, 2.0f)
            ),
            arrayOf(
                floatArrayOf(6.0f, 3.0f, 4.0f),
                floatArrayOf(7.0f, 2.0f, 5.0f)
            )
        )
        val layer = MaxPool1D(2, 2)

        EagerSession.create().use {
            val tf = Ops.create()
            val inputOp = tf.constant(input)
            val isTraining = tf.constant(true)
            val numberOfLosses = tf.constant(1.0f)
            val output = layer.build(tf, inputOp, isTraining, numberOfLosses).asOutput()

            // Check output shape is correct.
            val expectedShape = intArrayOf(input.size, 2, input[0][0].size)
            Assertions.assertArrayEquals(
                expectedShape,
                output.shape().toIntArray()
            )

            // Check output values are correct.
            val actual = Array(input.size) { Array(2) { FloatArray(input[0][0].size) } }
            output.tensor().copyTo(actual)
            for (i in expected.indices) {
                for (j in expected[i].indices) {
                    Assertions.assertArrayEquals(
                        expected[i][j],
                        actual[i][j]
                    )
                }
            }
        }
    }

    @Test
    fun withPaddingAndStride() {
        val input = arrayOf(
            arrayOf(
                floatArrayOf(1.0f, -2.0f, 3.0f),
                floatArrayOf(0.5f, 2.0f, 5.0f),
                floatArrayOf(-1.0f, 3.0f, 2.0f),
                floatArrayOf(1.5f, -1.0f, 0.5f)
            ),
            arrayOf(
                floatArrayOf(5.0f, 3.0f, 1.0f),
                floatArrayOf(6.0f, -2.5f, 4.0f),
                floatArrayOf(7.0f, 0.0f, 5.0f),
                floatArrayOf(1.0f, 2.0f, 4.0f)
            ),
        )
        val expected = arrayOf(
            arrayOf(
                floatArrayOf(1.0f, 2.0f, 5.0f),
                floatArrayOf(0.5f, 3.0f, 5.0f),
                floatArrayOf(1.5f, 3.0f, 2.0f),
                floatArrayOf(1.5f, -1.0f, 0.5f),
            ),
            arrayOf(
                floatArrayOf(6.0f, 3.0f, 4.0f),
                floatArrayOf(7.0f, 0.0f, 5.0f),
                floatArrayOf(7.0f, 2.0f, 5.0f),
                floatArrayOf(1.0f, 2.0f, 4.0f)
            )
        )
        val layer = MaxPool1D(strides = intArrayOf(1, 1, 1), padding = ConvPadding.SAME)

        EagerSession.create().use {
            val tf = Ops.create()
            val inputOp = tf.constant(input)
            val isTraining = tf.constant(true)
            val numberOfLosses = tf.constant(1.0f)
            val output = layer.build(tf, inputOp, isTraining, numberOfLosses).asOutput()

            // Check output shape is correct.
            val expectedShape = intArrayOf(input.size, input[0].size, input[0][0].size)
            Assertions.assertArrayEquals(
                expectedShape,
                output.shape().toIntArray()
            )

            // Check output values are correct.
            val actual = Array(input.size) { Array(input[0].size) { FloatArray(input[0][0].size) } }
            output.tensor().copyTo(actual)
            for (i in expected.indices) {
                for (j in expected[i].indices) {
                    Assertions.assertArrayEquals(
                        expected[i][j],
                        actual[i][j]
                    )
                }
            }
        }
    }

    @Test
    fun withPoolSizeAndStride() {
        val input = arrayOf(
            arrayOf(
                floatArrayOf(1.0f, -2.0f, 3.0f),
                floatArrayOf(0.5f, 2.0f, 5.0f),
                floatArrayOf(-1.0f, 3.0f, 2.0f),
                floatArrayOf(1.5f, -1.0f, 0.5f)
            ),
            arrayOf(
                floatArrayOf(5.0f, 3.0f, 1.0f),
                floatArrayOf(6.0f, -2.5f, 4.0f),
                floatArrayOf(7.0f, 0.0f, 5.0f),
                floatArrayOf(1.0f, 2.0f, 4.0f)
            ),
        )
        val expected = arrayOf(
            arrayOf(
                floatArrayOf(1.0f, 3.0f, 5.0f),
                floatArrayOf(1.5f, 3.0f, 5.0f),
            ),
            arrayOf(
                floatArrayOf(7.0f, 3.0f, 5.0f),
                floatArrayOf(7.0f, 2.0f, 5.0f),
            )
        )
        val layer = MaxPool1D(poolSize = intArrayOf(1, 3, 1), strides = intArrayOf(1, 1, 1))

        EagerSession.create().use {
            val tf = Ops.create()
            val inputOp = tf.constant(input)
            val isTraining = tf.constant(true)
            val numberOfLosses = tf.constant(1.0f)
            val output = layer.build(tf, inputOp, isTraining, numberOfLosses).asOutput()

            // Check output shape is correct.
            val expectedShape = intArrayOf(input.size, 2, input[0][0].size)
            Assertions.assertArrayEquals(
                expectedShape,
                output.shape().toIntArray()
            )

            // Check output values are correct.
            val actual = Array(input.size) { Array(2) { FloatArray(input[0][0].size) } }
            output.tensor().copyTo(actual)
            for (i in expected.indices) {
                for (j in expected[i].indices) {
                    Assertions.assertArrayEquals(
                        expected[i][j],
                        actual[i][j]
                    )
                }
            }
        }
    }
}
