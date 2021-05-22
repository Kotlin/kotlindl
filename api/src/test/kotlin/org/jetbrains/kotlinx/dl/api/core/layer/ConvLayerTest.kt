/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.activation.EPS
import org.jetbrains.kotlinx.dl.api.core.shape.*
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.tensorflow.*
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Constant

internal typealias FloatConv1DTensor = Array<Array<FloatArray>>

internal typealias FloatConv2DTensor = Array<Array<Array<FloatArray>>>

internal typealias AnyDTensor = Array<*>

open class ConvLayerTest {

    protected fun assertFloatConv1DTensorsEquals(
        layer: Layer,
        input: FloatConv1DTensor,
        expected: FloatConv1DTensor
    ) {
        val actual = expected.copyZeroed()
        assertTensorsEquals(layer, input, expected, actual,
            ::assertFloatConv1DTensorsEquals) { tf, tensor -> tf.constant(tensor.cast3DArray()) }
    }

    protected fun assertFloatConv2DTensorsEquals(
        layer: Layer,
        input: FloatConv2DTensor,
        expected: FloatConv2DTensor
    ) {
        val actual = expected.copyZeroed()
        assertTensorsEquals(layer, input, expected, actual,
            ::assertFloatConv2DTensorsEquals) { tf, tensor -> tf.constant(tensor.cast4DArray()) }
    }

    protected fun createFloatConv1DTensor(
        batchSize: Long,
        size: Long,
        channels: Long,
        initValue: Float
    ): FloatConv1DTensor =
        getFloatArrayOfShape(Shape.make(batchSize, size, channels), initValue).cast3DArray()

    protected fun createFloatConv2DTensor(
        batchSize: Long,
        height: Long,
        width: Long,
        channels: Long,
        initValue: Float
    ): FloatConv2DTensor =
        getFloatArrayOfShape(Shape.make(batchSize, height, width, channels), initValue).cast4DArray()

    private fun FloatConv1DTensor.copyZeroed(): FloatConv1DTensor =
        getFloatArrayOfShape(getShapeOfArray(this)).cast3DArray()

    private fun FloatConv2DTensor.copyZeroed(): FloatConv2DTensor =
        getFloatArrayOfShape(getShapeOfArray(this)).cast4DArray()

    private fun assertTensorsEquals(
        layer: Layer,
        input: AnyDTensor,
        expected: AnyDTensor,
        actual: AnyDTensor,
        assertEquals: (AnyDTensor, AnyDTensor) -> Unit,
        constProducer: (Ops, AnyDTensor) -> Constant<Float>
    ) {
        Graph().use { graph ->
            Session(graph).use { session ->
                val tf = Ops.create(graph)
                KGraph(graph.toGraphDef()).use { kGraph ->

                    val inputOp = constProducer(tf, input)
                    val isTraining = tf.constant(true)
                    val numberOfLosses = tf.constant(1.0f)

                    layer.build(tf, kGraph, getShapeOfArray(input))
                    val output = layer.forward(tf, inputOp, isTraining, numberOfLosses).asOutput()
                    kGraph.initializeGraphVariables(session)
                    val outputTensor = session.runner().fetch(output).run().first()
                    val outputTensorShape = shapeFromDims(*outputTensor.shape())
                    outputTensor.copyTo(actual)

                    assertEquals(getShapeOfArray(expected), outputTensorShape)
                    assertEquals(expected, actual)
                }
            }
        }
    }

    private fun assertFloatConv1DTensorsEquals(
        expected: AnyDTensor,
        actual: AnyDTensor
    ) {
        val expectedTensor = expected.cast3DArray()
        val actualTensor = actual.cast3DArray()
        val msg = "Expected ${expectedTensor.contentDeepToString()} " +
                "to equal ${actualTensor.contentDeepToString()}"
        for (i in expectedTensor.indices) {
            for (j in expectedTensor[i].indices) {
                assertArrayEquals(expectedTensor[i][j], actualTensor[i][j], EPS, msg)
            }
        }
    }

    private fun assertFloatConv2DTensorsEquals(
        expected: AnyDTensor,
        actual: AnyDTensor
    ) {
        val expectedTensor = expected.cast4DArray()
        val actualTensor = actual.cast4DArray()
        val msg = "Expected ${expectedTensor.contentDeepToString()} " +
                "to equal ${actualTensor.contentDeepToString()}"
        for (i in expectedTensor.indices) {
            for (j in expectedTensor[i].indices) {
                for (k in expectedTensor[i][j].indices) {
                    assertArrayEquals(expectedTensor[i][j][k], actualTensor[i][j][k], EPS, msg)
                }
            }
        }
    }
}
