/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.activation.EPS
import org.jetbrains.kotlinx.dl.api.core.shape.shapeFromDims
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.tensorflow.*
import org.tensorflow.op.Ops

internal typealias FloatConv2DTensor = Array<Array<Array<FloatArray>>>

open class ConvLayerTest {

    protected fun assertTensorsEquals(
        layer: Layer,
        input: FloatConv2DTensor,
        expected: FloatConv2DTensor
    ) {
        val actual = expected.copyZeroed()

        Graph().use { graph ->
            Session(graph).use { session ->
                val tf = Ops.create(graph)
                KGraph(graph.toGraphDef()).use { kGraph ->

                    val inputOp = tf.constant(input)
                    val isTraining = tf.constant(true)
                    val numberOfLosses = tf.constant(1.0f)

                    layer.build(tf, kGraph, input.getShape())
                    val output = layer.forward(tf, inputOp, isTraining, numberOfLosses).asOutput()
                    kGraph.initializeGraphVariables(session)
                    val outputTensor = session.runner().fetch(output).run().first()
                    val outputTensorShape = shapeFromDims(*outputTensor.shape())
                    outputTensor.copyTo(actual)

                    assertEquals(expected.getShape(), outputTensorShape)
                    assertTensorsEquals(expected, actual)
                }
            }
        }
    }

    protected fun assertTensorsEquals(
        expected: FloatConv2DTensor,
        actual: FloatConv2DTensor
    ) {
        for (i in expected.indices) {
            for (j in expected[i].indices) {
                for (k in expected[i][j].indices) {
                    assertArrayEquals(
                        expected[i][j][k],
                        actual[i][j][k],
                        EPS
                    )
                }
            }
        }
    }

    protected fun createFloatConv2DTensor(
        batchSize: Int,
        rows: Int,
        cols: Int,
        channels: (Pair<Int, Int>) -> FloatArray
    ): FloatConv2DTensor =
        Array(batchSize) {
            Array(rows) { r ->
                Array(cols) { c ->
                    channels(Pair(r, c))
                }
            }
        }

    protected fun allChannelsSame(channels: Int, value: Float): (Pair<Int, Int>) -> FloatArray =
        { FloatArray(channels) { value } }

    private fun FloatConv2DTensor.getShape(): Shape = Shape.make(
        this.size.toLong(),
        this[0].size.toLong(),
        this[0][0].size.toLong(),
        this[0][0][0].size.toLong()
    )

    private fun FloatConv2DTensor.copyZeroed() =
        createFloatConv2DTensor(this.size, this[0].size, this[0][0].size) {
            FloatArray(this[0][0][0].size)
        }
}
