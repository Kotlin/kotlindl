/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.metric

import org.jetbrains.kotlinx.dl.api.core.activation.ActivationTest
import org.jetbrains.kotlinx.dl.api.core.activation.EPS
import org.jetbrains.kotlinx.dl.api.core.activation.SigmoidActivation
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import org.tensorflow.EagerSession
import org.tensorflow.Operand
import org.tensorflow.op.Ops

internal class AccuracyTest : ActivationTest() {
    @Test
    fun apply() {
        val input = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f)
        val actual = floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
        val expected = floatArrayOf(
            0.7310586f, 0.8807971f, 0.95257413f, 0.98201376f, 0.9933071f,
            0.99752736f, 0.999089f, 0.99966455f, 0.9998766f, 0.9999546f
        )

        EagerSession.create().use { session ->
            val tf = Ops.create(session)
            val instance = SigmoidActivation()
            val operand: Operand<Float> = instance.apply(tf, tf.constant(input))
            operand.asOutput().tensor().copyTo(actual)

            assertArrayEquals(
                expected,
                actual,
                EPS
            )
        }
    }
}