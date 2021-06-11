package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.MaxPool3D
import org.jetbrains.kotlinx.dl.api.core.loss.EPS
import org.jetbrains.kotlinx.dl.api.core.shape.shapeFromDims
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import org.tensorflow.EagerSession
import org.tensorflow.Graph
import org.tensorflow.op.Ops
import org.tensorflow.Shape
import org.junit.jupiter.api.Assertions.assertEquals

internal class MaxPool3DTest {

    private val inputArr = Array(1) {
        Array(
            30
        ) {
            Array(30) {
                Array(30) {
                    FloatArray(3, { 0f })
                }
            }
        }
    }

    private val inputNumeric = arrayOf(
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

    private fun assertMaxPool3D(
        layer: Layer,
        expected: Array<Array<Array<Array<FloatArray>>>>,
        input: Array<Array<Array<Array<FloatArray>>>> = inputArr
    ) {
        val inputShape: Shape = Shape.make(
            input.size.toLong(),
            input[0].size.toLong(),
            input[0][0].size.toLong(),
            input[0][0][0].size.toLong(),
            input[0][0][0][0].size.toLong(),
        )
        EagerSession.create().use {
            val tf = Ops.create()
            layer.build(tf, KGraph(Graph().toGraphDef()), inputShape)
            val inputOp = tf.constant(input)
            val isTraining = tf.constant(true)
            val numOfLosses = tf.constant(1.0f)
            val output = layer.forward(tf, inputOp, isTraining, numOfLosses).asOutput().tensor()

            val expectedShape = Shape.make(
                expected.size.toLong(),
                expected[0].size.toLong(),
                expected[0][0].size.toLong(),
                expected[0][0][0].size.toLong(),
                expected[0][0][0][0].size.toLong(),
            )

            val actualShape = shapeFromDims(*output.shape())
            assertEquals(expectedShape, actualShape)

            val actual = Array(expected.size) {
                Array(expected[0].size) {
                    Array(expected[0][0].size) {
                        Array(expected[0][0][0].size) {
                            FloatArray(
                                expected[0][0][0][0].size
                            )
                        }
                    }
                }
            }
            output.copyTo(actual)
            for (i in expected.indices) {
                for (j in expected[i].indices) {
                    for (k in expected[i][j].indices) {
                        for (l in expected[i][j][k].indices) {
                            assertArrayEquals(expected[i][j][k][l], actual[i][j][k][l], EPS)
                        }
                    }
                }
            }
        }
    }

    @Test
    fun defaultTest() {
        var expected = Array(1) {
            Array(
                15
            ) {
                Array(15) {
                    Array(15) {
                        FloatArray(3, { 0f })
                    }
                }
            }
        }

        val layer = MaxPool3D()
        assertMaxPool3D(layer, expected)
    }

    @Test
    fun poolSizeTest() {
        var expected = Array(1) {
            Array(
                14
            ) {
                Array(14) {
                    Array(14) {
                        FloatArray(3, { 0f })
                    }
                }
            }
        }
        val layer = MaxPool3D(poolSize = intArrayOf(1, 3, 3, 3, 1))
        assertMaxPool3D(layer, expected)
    }

    @Test
    fun strideSizeTest() {
        var expected = Array(1) {
            Array(
                8
            ) {
                Array(8) {
                    Array(8) {
                        FloatArray(3, { 0f })
                    }
                }
            }
        }
        val layer = MaxPool3D(strides = intArrayOf(1, 4, 4, 4, 1))
        assertMaxPool3D(layer, expected)
    }

    @Test
    fun defaultNumericInputTest() {
        val expected = arrayOf(
            arrayOf(
                arrayOf(
                    arrayOf(
                        floatArrayOf(7.0f, 3.0f, 5.0f),
                        floatArrayOf(7.0f, 5.0f, 5.0f),
                    ),
                ),
            ),
        )
        val layer = MaxPool3D()
        assertMaxPool3D(layer, expected, inputNumeric)
    }

    @Test
    fun poolSizeNumericInputTest() {
        val expected = arrayOf(
            arrayOf(
                arrayOf(
                    arrayOf(
                        floatArrayOf(7.0f, 5.0f, 5.0f),
                    ),
                ),
            ),
        )
        val layer = MaxPool3D(poolSize = intArrayOf(1, 2, 2, 3, 1))
        assertMaxPool3D(layer, expected, inputNumeric)
    }

    @Test
    fun strideNumericInputTest() {
        val expected = arrayOf(
            arrayOf(
                arrayOf(
                    arrayOf(
                        floatArrayOf(7.0f, 3.0f, 5.0f),
                    ),
                ),
            ),
        )
        val layer = MaxPool3D(strides = intArrayOf(1, 3, 3, 3, 1))
        assertMaxPool3D(layer, expected, inputNumeric)
    }
}
