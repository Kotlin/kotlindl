/*
 * Copyright 2021-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.RepeatVector
import org.jetbrains.kotlinx.dl.api.core.shape.toIntArray
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test
import org.tensorflow.Output
import org.tensorflow.op.Ops

internal class RepeatVectorLayerTest {

    @Test
    fun testIllegalRepetitions() {
        Assertions.assertThrows(IllegalArgumentException::class.java) {
            RepeatVector(n = -10)
        }
    }

    @Test
    fun testOutputShape() {
        val layer = RepeatVector(n = 2)
        val x = Array(10) { FloatArray(10) { 1F } }
        val y = layer(x)
        Assertions.assertArrayEquals(intArrayOf(10, layer.n, 10), y.shape().toIntArray())
    }

    @Test
    fun testOutput() {
        val layer = RepeatVector(n = 2)
        val x = Array(3) { FloatArray(3) { it.toFloat() } }
        val y = layer(x)
        val actual = y.tensor().copyTo(Array(3) { Array(layer.n) { FloatArray(3) } })
        val expected = arrayOf(
            arrayOf(floatArrayOf(0F, 1F, 2F), floatArrayOf(0F, 1F, 2F)),
            arrayOf(floatArrayOf(0F, 1F, 2F), floatArrayOf(0F, 1F, 2F)),
            arrayOf(floatArrayOf(0F, 1F, 2F), floatArrayOf(0F, 1F, 2F))
        )
        Assertions.assertArrayEquals(expected, actual)
    }

    // TODO: generalise this for Layer, see https://github.com/JetBrains/KotlinDL/issues/145
    private operator fun RepeatVector.invoke(input: Array<FloatArray>): Output<Float> = Ops.create().let { tf ->
        val inputOp = tf.constant(input)
        val isTraining = tf.constant(true)
        val numberOfLosses = tf.constant(1.0f)
        build(tf, inputOp, isTraining, numberOfLosses).asOutput()
    }
}
