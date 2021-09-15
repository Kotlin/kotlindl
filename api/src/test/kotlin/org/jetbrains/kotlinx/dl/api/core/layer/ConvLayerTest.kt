/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.activation.EPS
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.AbstractConv
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv1D
import org.jetbrains.kotlinx.dl.api.core.shape.*
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.tensorflow.*
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Constant
import java.lang.IllegalArgumentException

internal typealias FloatConv1DTensor = Array<Array<FloatArray>>

internal typealias FloatConv2DTensor = Array<Array<Array<FloatArray>>>

internal typealias FloatConv3DTensor = Array<Array<Array<Array<FloatArray>>>>

internal typealias AnyDTensor = Array<*>

open class ConvLayerTest {

    protected fun assertFloatConv1DTensorsEquals(
        layer: AbstractConv,
        input: FloatConv1DTensor,
        expected: FloatConv1DTensor
    ) {
        val actual = expected.copyZeroed()
        assertTensorsEquals(
            layer, input, expected, actual,
            ::assertFloatConv1DTensorsEquals
        ) { tf, tensor -> tf.constant(tensor.cast3D<FloatArray>()) }
    }

    protected fun assertFloatConv2DTensorsEquals(
        layer: AbstractConv,
        input: FloatConv2DTensor,
        expected: FloatConv2DTensor
    ) {
        val actual = expected.copyZeroed()
        assertTensorsEquals(
            layer, input, expected, actual,
            ::assertFloatConv2DTensorsEquals
        ) { tf, tensor -> tf.constant(tensor.cast4D<FloatArray>()) }
    }

    protected fun assertFloatConv3DTensorsEquals(
        layer: AbstractConv,
        input: FloatConv3DTensor,
        expected: FloatConv3DTensor
    ) {
        val actual = expected.copyZeroed()
        assertTensorsEquals(
            layer, input, expected, actual,
            ::assertFloatConv3DTensorsEquals
        ) { tf, tensor -> tf.constant(tensor.cast5D<FloatArray>()) }
    }

    protected fun createFloatConv1DTensor(
        batchSize: Long,
        size: Long,
        channels: Long,
        initValue: Float
    ): FloatConv1DTensor =
        getFloatArrayOfShape(Shape.make(batchSize, size, channels), initValue).cast3D()

    protected fun createFloatConv2DTensor(
        batchSize: Long,
        height: Long,
        width: Long,
        channels: Long,
        initValue: Float
    ): FloatConv2DTensor =
        getFloatArrayOfShape(Shape.make(batchSize, height, width, channels), initValue).cast4D()

    protected fun createFloatConv3DTensor(
        batchSize: Long,
        depth: Long,
        height: Long,
        width: Long,
        channels: Long,
        initValue: Float
    ): FloatConv3DTensor =
        getFloatArrayOfShape(Shape.make(batchSize, depth, height, width, channels), initValue).cast5D()

    private fun FloatConv1DTensor.copyZeroed(): FloatConv1DTensor =
        getFloatArrayOfShape(this.shape).cast3D()

    private fun FloatConv2DTensor.copyZeroed(): FloatConv2DTensor =
        getFloatArrayOfShape(this.shape).cast4D()

    private fun FloatConv3DTensor.copyZeroed(): FloatConv3DTensor =
        getFloatArrayOfShape(this.shape).cast5D()

    private fun assertTensorsEquals(
        layer: AbstractConv,
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

                    layer.build(tf, kGraph, input.shape)
                    val output = layer.forward(tf, inputOp, isTraining, numberOfLosses).asOutput()

                    layer.initialize(session)

                    val outputTensor = session.runner().fetch(output).run().first()
                    val outputTensorShape = shapeFromDims(*outputTensor.shape())
                    outputTensor.copyTo(actual)

                    assertEquals(expected.shape, outputTensorShape)
                    assertEquals(expected, actual)
                }
            }
        }
    }

    private fun assertFloatConv1DTensorsEquals(
        expected: AnyDTensor,
        actual: AnyDTensor
    ) {
        val expectedTensor = expected.cast3D<FloatArray>()
        val actualTensor = actual.cast3D<FloatArray>()
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
        val expectedTensor = expected.cast4D<FloatArray>()
        val actualTensor = actual.cast4D<FloatArray>()
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

    private fun assertFloatConv3DTensorsEquals(
        expected: AnyDTensor,
        actual: AnyDTensor
    ) {
        val expectedTensor = expected.cast5D<FloatArray>()
        val actualTensor = actual.cast5D<FloatArray>()
        val msg = "Expected ${expectedTensor.contentDeepToString()} " +
                "to equal ${actualTensor.contentDeepToString()}"
        for (i in expectedTensor.indices) {
            for (j in expectedTensor[i].indices) {
                for (k in expectedTensor[i][j].indices) {
                    for (l in expectedTensor[i][j][k].indices) {
                        assertArrayEquals(expectedTensor[i][j][k][l], actualTensor[i][j][k][l], EPS, msg)
                    }
                }
            }
        }
    }

    protected fun Array<*>.sum(): Float = fold(0.0f) { acc, arr ->
        when (arr) {
            is FloatArray -> arr.sum() + acc
            is Array<*> -> arr.sum() + acc
            else -> throw IllegalArgumentException("Cannot sum array other than Array of FloatArray")
        }
    }
}
