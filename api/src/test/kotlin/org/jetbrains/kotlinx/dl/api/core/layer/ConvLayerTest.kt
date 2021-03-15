/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.activation.EPS
import org.jetbrains.kotlinx.dl.api.core.shape.shapeFromDims
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Assertions.assertEquals
import org.tensorflow.Graph
import org.tensorflow.Session
import org.tensorflow.Shape
import org.tensorflow.op.Ops

// TODO: tests should start from variable initialization in Graph mode only. Looks like eager mode does not support variables
// We could tests without variables only
open class ConvLayerTest {
    protected fun assertActivationFunction(
        layer: Layer,
        input: Array<Array<Array<FloatArray>>>,
        actual: Array<Array<Array<FloatArray>>>,
        expected: Array<Array<Array<FloatArray>>>
    ) {
        val inputSize = input.size
        val inputShape = Shape.make(inputSize.toLong())

        Graph().use { g ->
            Session(g).use { session ->
                val tf = Ops.create(g)
                val kGraph = KGraph(g.toGraphDef())
                val inputOp = tf.constant(input)


                layer.build(tf, kGraph, inputShape)

                val isTraining = tf.constant(true)
                val numberOfLosses = tf.constant(1.0f)
                val output = layer.forward(tf, inputOp, isTraining, numberOfLosses).asOutput().tensor()

                val expectedShape = Shape.make(
                    inputSize.toLong()
                )

                val actualShape = shapeFromDims(*output.shape())
                assertEquals(expectedShape, actualShape)

                output.copyTo(actual)

                for (i in expected.indices) {
                    for (j in expected[i].indices) {
                        for (k in expected[i][j].indices) {
                            Assertions.assertArrayEquals(
                                expected[i][j][k],
                                actual[i][j][k],
                                EPS
                            )
                        }

                    }
                }

            }
        }



       /* EagerSession.create().use {
            val tf = Ops.create(it)
            val inputOp = tf.constant(input)
            layer.build(tf, KGraph(Graph().toGraphDef()), inputShape)
            val isTraining = tf.constant(true)
            val numberOfLosses = tf.constant(1.0f)
            val output = layer.forward(tf, inputOp, isTraining, numberOfLosses).asOutput().tensor()

            val expectedShape = Shape.make(
                inputSize.toLong()
            )

            val actualShape = shapeFromDims(*output.shape())
            assertEquals(expectedShape, actualShape)

            output.copyTo(actual)

            for (i in expected.indices) {
                for (j in expected[i].indices) {
                    for (k in expected[i][j].indices) {
                        Assertions.assertArrayEquals(
                            expected[i][j][k],
                            actual[i][j][k],
                            EPS
                        )
                    }

                }
            }
        }*/
    }
}

