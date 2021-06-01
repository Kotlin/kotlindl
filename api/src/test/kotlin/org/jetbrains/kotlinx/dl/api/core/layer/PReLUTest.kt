/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.activation.EPS
import org.jetbrains.kotlinx.dl.api.core.initializer.Constant
import org.jetbrains.kotlinx.dl.api.core.layer.activation.PReLU
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.shape.shapeFromDims
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test
import org.tensorflow.EagerSession
import org.tensorflow.Graph
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

        EagerSession.create().use {
            val tf = Ops.create(it)
            val graph = KGraph(Graph().toGraphDef())

            val inputShape = Shape.make(input.size.toLong(), input[0].size.toLong())
            layer.build(tf, graph, inputShape)
            val inputOp = tf.constant(input)
            val isTraining = tf.constant(false)
            val numberOfLosses = tf.constant(1.0f)
            val output = layer.forward(tf, inputOp, isTraining, numberOfLosses).asOutput().tensor()

            // Check the output shape is correct.
            val expectedShape = inputShape
            val actualShape = shapeFromDims(*output.shape())
            Assertions.assertEquals(expectedShape, actualShape)

            // Check output is correct.
            val actual = Array(input.size) { FloatArray(input[0].size) }
            output.copyTo(actual)
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