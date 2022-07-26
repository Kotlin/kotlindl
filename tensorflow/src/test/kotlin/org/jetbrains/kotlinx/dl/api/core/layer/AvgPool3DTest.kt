/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.AvgPool3D
import org.jetbrains.kotlinx.dl.api.core.shape.shape
import org.jetbrains.kotlinx.dl.api.core.shape.toIntArray
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test
import org.tensorflow.EagerSession
import org.tensorflow.Shape
import org.tensorflow.op.Ops

private const val EPS: Float = 1e-6f

internal class AvgPool3DTest {

    private val input = arrayOf(
        arrayOf(
            arrayOf(
                arrayOf(
                    floatArrayOf(1.0f, -2.0f, 3.0f),
                    floatArrayOf(0.5f, 2.0f, 5.0f),
                    floatArrayOf(-1.0f, 3.0f, 2.0f),
                    floatArrayOf(1.5f, -1.0f, 0.5f)
                ),
                arrayOf(
                    floatArrayOf(-1.0f, 2.0f, -2.0f),
                    floatArrayOf(2.5f, 3.0f, 1.0f),
                    floatArrayOf(-2.0f, 3.0f, 2.5f),
                    floatArrayOf(-3.0f, 1.0f, 1.5f)
                ),
            ),
            arrayOf(
                arrayOf(
                    floatArrayOf(1.0f, 3.0f, 1.0f),
                    floatArrayOf(6.0f, -2.5f, 4.0f),
                    floatArrayOf(7.0f, 0.0f, 5.0f),
                    floatArrayOf(1.0f, 2.0f, 4.0f)
                ),
                arrayOf(
                    floatArrayOf(7.0f, -3.0f, 2.0f),
                    floatArrayOf(1.0f, 2.0f, 2.0f),
                    floatArrayOf(3.0f, 5.0f, -2.0f),
                    floatArrayOf(3.0f, -1.0f, 0.0f)
                ),
            ),
        ),
    )

    private val inputShape: Shape = input.shape

    @Test
    fun default() {
        val layer = AvgPool3D(2, 2)
        val expected = arrayOf(
            arrayOf(
                arrayOf(
                    arrayOf(
                        floatArrayOf(18.0f / 8, 4.5f / 8, 16.0f / 8),
                        floatArrayOf(9.5f / 8, 12.0f / 8, 13.5f / 8),
                    ),
                ),
            ),
        )

        EagerSession.create().use {
            val tf = Ops.create()
            val inputOp = tf.constant(input)
            val isTraining = tf.constant(true)
            val numberOfLosses = tf.constant(1.0f)
            val output = layer.build(tf, inputOp, isTraining, numberOfLosses).asOutput()

            // Check output shape is correct.
            val expectedShape = intArrayOf(input.size, 1, 1, 2, input[0][0][0][0].size)
            Assertions.assertArrayEquals(
                expectedShape,
                output.shape().toIntArray()
            )

            // Check output values are correct.
            val actual = Array(input.size) {
                Array(1) { Array(1) { Array(2) { FloatArray(input[0][0][0][0].size) } } }
            }
            output.tensor().copyTo(actual)
            for (i in expected.indices) {
                for (j in expected[i].indices) {
                    for (k in expected[i][j].indices) {
                        for (l in expected[i][j][k].indices) {
                            Assertions.assertArrayEquals(
                                expected[i][j][k][l],
                                actual[i][j][k][l],
                                EPS
                            )
                        }
                    }
                }
            }
        }
    }

    @Test
    fun withPaddingAndStride() {
        val layer = AvgPool3D(strides = intArrayOf(1, 1, 1, 1, 1), padding = ConvPadding.SAME)
        val expected = arrayOf(
            arrayOf(
                arrayOf(
                    arrayOf(
                        floatArrayOf(18.0f / 8, 4.5f / 8, 16.0f / 8),
                        floatArrayOf(17.0f / 8, 15.5f / 8, 19.5f / 8),
                        floatArrayOf(9.5f / 8, 12.0f / 8, 13.5f / 8),
                        floatArrayOf(5.0f / 8, 2.0f / 8, 12.0f / 8)
                    ),
                    arrayOf(
                        floatArrayOf(19.0f / 8, 8.0f / 8, 6.0f / 8),
                        floatArrayOf(9.0f / 8, 26.0f / 8, 7.0f / 8),
                        floatArrayOf(2.0f / 8, 16.0f / 8, 4.0f / 8),
                        floatArrayOf(0.0f / 8, 0.0f / 8, 6.0f / 8)
                    ),
                ),
                arrayOf(
                    arrayOf(
                        floatArrayOf(30.0f / 8, -1.0f / 8, 18.0f / 8),
                        floatArrayOf(34.0f / 8, 9.0f / 8, 18.0f / 8),
                        floatArrayOf(28.0f / 8, 12.0f / 8, 14.0f / 8),
                        floatArrayOf(16.0f / 8, 4.0f / 8, 16.0f / 8)
                    ),
                    arrayOf(
                        floatArrayOf(32.0f / 8, -4.0f / 8, 16.0f / 8),
                        floatArrayOf(16.0f / 8, 28.0f / 8, 0.0f / 8),
                        floatArrayOf(24.0f / 8, 16.0f / 8, -8.0f / 8),
                        floatArrayOf(24.0f / 8, -8.0f / 8, 0.0f / 8)
                    ),
                ),
            ),
        )

        EagerSession.create().use {
            val tf = Ops.create()
            val inputOp = tf.constant(input)
            val isTraining = tf.constant(true)
            val numberOfLosses = tf.constant(1.0f)
            val output = layer.build(tf, inputOp, isTraining, numberOfLosses).asOutput()

            // Check output shape is correct.
            val expectedShape = inputShape.toIntArray()
            Assertions.assertArrayEquals(
                expectedShape,
                output.shape().toIntArray()
            )

            // Check output values are correct.
            val actual = Array(input.size) {
                Array(input[0].size) {
                    Array(input[0][0].size) {
                        Array(input[0][0][0].size) {
                            FloatArray(input[0][0][0][0].size)
                        }
                    }
                }
            }
            output.tensor().copyTo(actual)
            for (i in expected.indices) {
                for (j in expected[i].indices) {
                    for (k in expected[i][j].indices) {
                        for (l in expected[i][j][k].indices) {
                            Assertions.assertArrayEquals(
                                expected[i][j][k][l],
                                actual[i][j][k][l],
                                EPS
                            )
                        }
                    }
                }
            }
        }
    }

    @Test
    fun withPoolSizeAndStride() {
        val layer = AvgPool3D(poolSize = intArrayOf(1, 2, 2, 3, 1), strides = intArrayOf(1, 1, 1, 1, 1))
        val expected = arrayOf(
            arrayOf(
                arrayOf(
                    arrayOf(
                        floatArrayOf(25.0f / 12, 15.5f / 12, 23.5f / 12),
                        floatArrayOf(19.5f / 12, 16.5f / 12, 25.5f / 12),
                    ),
                ),
            ),
        )

        EagerSession.create().use {
            val tf = Ops.create()
            val inputOp = tf.constant(input)
            val isTraining = tf.constant(true)
            val numberOfLosses = tf.constant(1.0f)
            val output = layer.build(tf, inputOp, isTraining, numberOfLosses).asOutput()

            // Check output shape is correct.
            val expectedShape = intArrayOf(input.size, 1, 1, 2, input[0][0][0][0].size)
            Assertions.assertArrayEquals(
                expectedShape,
                output.shape().toIntArray()
            )

            // Check output values are correct.
            val actual = Array(input.size) {
                Array(1) { Array(1) { Array(2) { FloatArray(input[0][0][0][0].size) } } }
            }
            output.tensor().copyTo(actual)
            for (i in expected.indices) {
                for (j in expected[i].indices) {
                    for (k in expected[i][j].indices) {
                        for (l in expected[i][j][k].indices) {
                            Assertions.assertArrayEquals(
                                expected[i][j][k][l],
                                actual[i][j][k][l],
                                EPS
                            )
                        }
                    }
                }
            }
        }
    }
}
