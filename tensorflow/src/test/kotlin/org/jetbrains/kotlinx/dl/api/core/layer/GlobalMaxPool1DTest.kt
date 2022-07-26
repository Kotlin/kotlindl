/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.layer.pooling.GlobalMaxPool1D
import org.jetbrains.kotlinx.dl.api.core.shape.toIntArray
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test
import org.tensorflow.EagerSession
import org.tensorflow.op.Ops

internal class GlobalMaxPool1DTest {
    @Test
    fun default() {
        val input = arrayOf(
            arrayOf(
                floatArrayOf(1.0f, -2.0f, 3.0f),
                floatArrayOf(0.5f, 2.0f, 5.0f)
            ),
            arrayOf(
                floatArrayOf(5.0f, 3.0f, 1.0f),
                floatArrayOf(6.0f, -2.5f, 4.0f),
            )
        )
        val expected = arrayOf(
            floatArrayOf(1.0f, 2.0f, 5.0f),
            floatArrayOf(6.0f, 3.0f, 4.0f)
        )
        val layer = GlobalMaxPool1D()

        EagerSession.create().use {
            val tf = Ops.create()
            val inputOp = tf.constant(input)
            val isTraining = tf.constant(true)
            val numberOfLosses = tf.constant(1.0f)
            val output = layer.build(tf, inputOp, isTraining, numberOfLosses).asOutput()

            // Check output shape is correct.
            val expectedShape = intArrayOf(input.size, input[0][0].size)
            Assertions.assertArrayEquals(
                expectedShape,
                output.shape().toIntArray()
            )

            // Check output values are correct.
            val actual = Array(input.size) { FloatArray(input[0][0].size) }
            output.tensor().copyTo(actual)
            for (i in expected.indices) {
                Assertions.assertArrayEquals(
                    expected[i],
                    actual[i]
                )
            }
        }
    }
}
