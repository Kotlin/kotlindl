/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.activation.EPS
import org.jetbrains.kotlinx.dl.api.core.initializer.Constant
import org.jetbrains.kotlinx.dl.api.core.initializer.RandomUniform
import org.jetbrains.kotlinx.dl.api.core.layer.activation.PReLU
import org.jetbrains.kotlinx.dl.api.core.shape.shapeFromDims
import org.jetbrains.kotlinx.dl.api.core.shape.shapeOperand
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test
import org.tensorflow.Graph
import org.tensorflow.Session
import org.tensorflow.Shape
import org.tensorflow.op.Ops

class PReLUTest : ActivationLayerTest() {
    @Test
    fun default() {
        val input = arrayOf(
            floatArrayOf(-3.0f, -1.0f, 0.0f, 2.0f),
            floatArrayOf(2.5f, 1.0f, -5.0f, 4.2f),
        )
        val expected = arrayOf(
            floatArrayOf(0.0f, 0.0f, 0.0f, 2.0f),
            floatArrayOf(2.5f, 1.0f, 0.0f, 4.2f),
        )
        val layer = PReLU()

        Graph().use { graph ->
            Session(graph).use { session ->
                val tf = Ops.create(graph)
                KGraph(graph.toGraphDef()).use { kGraph ->
                    val inputShape = Shape.make(input.size.toLong(), input[0].size.toLong())
                    layer.build(tf, kGraph, inputShape)

                    val inputOp = tf.constant(input)
                    val isTraining = tf.constant(false)
                    val numberOfLosses = tf.constant(1.0f)
                    val output = layer.forward(tf, inputOp, isTraining, numberOfLosses).asOutput()
                    kGraph.initializeGraphVariables(session)
                    val outputTensor = session.runner().fetch(output).run().first()

                    // Check the output shape is correct.
                    val expectedShape = inputShape
                    val actualShape = shapeFromDims(*outputTensor.shape())
                    Assertions.assertEquals(expectedShape, actualShape)

                    // Check output is correct.
                    val actual = Array(input.size) { FloatArray(input[0].size) }
                    outputTensor.copyTo(actual)
                    for (i in actual.indices) {
                        Assertions.assertArrayEquals(
                            expected[i],
                            actual[i],
                            EPS
                        )
                    }
                }
            }
        }
    }

    @Test
    fun withInitializer() {
        val input = arrayOf(
            floatArrayOf(-3.0f, -1.0f, 0.0f, 2.0f),
            floatArrayOf(2.5f, 1.0f, -5.0f, 4.2f),
        )
        val expected = arrayOf(
            floatArrayOf(-6.0f, -2.0f, 0.0f, 2.0f),
            floatArrayOf(2.5f, 1.0f, -10.0f, 4.2f),
        )
        val layer = PReLU(Constant(2.0f))

        Graph().use { graph ->
            Session(graph).use { session ->
                val tf = Ops.create(graph)
                KGraph(graph.toGraphDef()).use { kGraph ->
                    val inputShape = Shape.make(input.size.toLong(), input[0].size.toLong())
                    layer.build(tf, kGraph, inputShape)

                    val inputOp = tf.constant(input)
                    val isTraining = tf.constant(false)
                    val numberOfLosses = tf.constant(1.0f)
                    val output = layer.forward(tf, inputOp, isTraining, numberOfLosses).asOutput()
                    kGraph.initializeGraphVariables(session)
                    val outputTensor = session.runner().fetch(output).run().first()

                    // Check the output shape is correct.
                    val expectedShape = inputShape
                    val actualShape = shapeFromDims(*outputTensor.shape())
                    Assertions.assertEquals(expectedShape, actualShape)

                    // Check output is correct.
                    val actual = Array(input.size) { FloatArray(input[0].size) }
                    outputTensor.copyTo(actual)
                    for (i in actual.indices) {
                        Assertions.assertArrayEquals(
                            expected[i],
                            actual[i],
                            EPS
                        )
                    }
                }
            }
        }
    }

    @Test
    fun withSharedAxes() {
        val input = arrayOf(
            arrayOf(
                floatArrayOf(-3.0f, -1.0f, 0.0f, 2.0f),
                floatArrayOf(2.5f, 1.0f, -5.0f, 4.2f),
            ),
            arrayOf(
                floatArrayOf(2.0f, 0.5f, 1.0f, 4.0f),
                floatArrayOf(-3.0f, -1.0f, 5.0f, -0.3f),
            )
        )
        val initializerSeed = 42L
        val layer = PReLU(RandomUniform(seed = initializerSeed), sharedAxes = intArrayOf(1))

        Graph().use { graph ->
            Session(graph).use { session ->
                val tf = Ops.create(graph)
                KGraph(graph.toGraphDef()).use { kGraph ->
                    val inputShape = Shape.make(
                        input.size.toLong(), input[0].size.toLong(), input[0][0].size.toLong()
                    )
                    layer.build(tf, kGraph, inputShape)

                    val inputOp = tf.constant(input)
                    val isTraining = tf.constant(false)
                    val numberOfLosses = tf.constant(1.0f)
                    val output = layer.forward(tf, inputOp, isTraining, numberOfLosses).asOutput()
                    kGraph.initializeGraphVariables(session)
                    val outputTensor = session.runner().fetch(output).run().first()

                    // Check the output shape is correct.
                    val expectedShape = inputShape
                    val actualShape = shapeFromDims(*outputTensor.shape())
                    Assertions.assertEquals(expectedShape, actualShape)

                    // TODO: find a better way to verify the output is correct.
                    // Check output is correct.
                    val alphaInitializer = RandomUniform(seed = initializerSeed)
                    val alphaTensor = session.runner().fetch(
                        alphaInitializer.initialize(
                            input[0][0].size,
                            input[0][0].size,
                            tf,
                            shapeOperand(tf, Shape.make(1, input[0][0].size.toLong())),
                            "temp_init"
                        ).asOutput()
                    ).run().first()
                    val alpha = Array(1) { FloatArray(input[0][0].size) }
                    alphaTensor.copyTo(alpha)

                    val actual = Array(input.size) {
                        Array(input[0].size) {
                            FloatArray(input[0][0].size)
                        }
                    }
                    outputTensor.copyTo(actual)
                    for (i in actual.indices) {
                        for (j in actual[i].indices) {
                            for (k in actual[i][j].indices) {
                                val it = input[i][j][k]
                                Assertions.assertEquals(
                                    if ( it < 0) it * alpha[0][k] else it,
                                    actual[i][j][k],
                                    EPS
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}