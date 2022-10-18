/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.initializer.Constant
import org.jetbrains.kotlinx.dl.api.core.initializer.RandomUniform
import org.jetbrains.kotlinx.dl.api.core.layer.activation.PReLU
import org.jetbrains.kotlinx.dl.api.core.shape.shape
import org.jetbrains.kotlinx.dl.api.core.shape.shapeOperand
import org.jetbrains.kotlinx.dl.api.core.shape.toLongArray
import org.junit.jupiter.api.Test
import org.tensorflow.EagerSession
import org.tensorflow.Shape
import org.tensorflow.op.Ops

class PReLUTest : LayerTest() {

    private val input = arrayOf(
        floatArrayOf(-3.0f, -1.0f, 0.0f, 2.0f),
        floatArrayOf(2.5f, 1.0f, -5.0f, 4.2f),
    )

    private val inputShape = input.shape.toLongArray()

    @Test
    fun default() {
        val expected = arrayOf(
            floatArrayOf(0.0f, 0.0f, 0.0f, 2.0f),
            floatArrayOf(2.5f, 1.0f, 0.0f, 4.2f),
        )
        val layer = PReLU()

        assertLayerOutputIsCorrect(layer, input, expected, RunMode.GRAPH)
        val expectedShape = inputShape
        assertLayerComputedOutputShape(layer, expectedShape)
    }

    @Test
    fun withInitializer() {
        val expected = arrayOf(
            floatArrayOf(-6.0f, -2.0f, 0.0f, 2.0f),
            floatArrayOf(2.5f, 1.0f, -10.0f, 4.2f),
        )
        val layer = PReLU(Constant(2.0f))
        assertLayerOutputIsCorrect(layer, input, expected, RunMode.GRAPH)
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
        val inputShape = input.shape.toLongArray()
        val initializerSeed = 42L
        val layer = PReLU(RandomUniform(seed = initializerSeed), sharedAxes = intArrayOf(1))

        val alpha = Array(1) { FloatArray(inputShape[2].toInt()) }
        val alphaInitializer = RandomUniform(seed = initializerSeed)
        EagerSession.create().use {
            val tf = Ops.create()
            val alphaTensor = alphaInitializer.initialize(
                inputShape[2].toInt(),
                inputShape[2].toInt(),
                tf,
                shapeOperand(tf, Shape.make(1, inputShape[2])),
                "temp_init"
            ).asOutput().tensor()
            alphaTensor.copyTo(alpha)
            alphaTensor.close()
        }

        val expected = Array(inputShape[0].toInt()) { i ->
            Array(inputShape[1].toInt()) { j ->
                FloatArray(inputShape[2].toInt()) { k ->
                    val it = input[i][j][k]
                    if (it < 0) it * alpha[0][k] else it
                }
            }
        }

        assertLayerOutputIsCorrect(layer, input, expected, RunMode.GRAPH)
        val expectedShape = inputShape
        assertLayerComputedOutputShape(layer, expectedShape)
    }
}
