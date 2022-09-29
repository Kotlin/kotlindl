/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.initializer

import org.jetbrains.kotlinx.dl.api.core.shape.shapeOperand
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import org.tensorflow.EagerSession
import org.tensorflow.Shape
import org.tensorflow.op.Ops

private const val EPS = 1e-7f
private const val FAN_IN = 10
private const val FAN_OUT = 20
private const val MEAN = 0.0f
private const val STD_DEV = 3.0f
private const val P1 = -4f
private const val P2 = 4f

internal class ParametrizedRandomNormalTest {
    @Test
    fun initialize() {
        val actual = Array(2) { FloatArray(2) { 0f } }
        val expected = Array(2) { FloatArray(2) { 0f } }
        expected[0][0] = -1.366183f
        expected[0][1] = 1.152605f
        expected[1][0] = 2.1622126f
        expected[1][1] = 3.8308368f

        val shape = Shape.make(2, 2)

        EagerSession.create().use { session ->
            val tf = Ops.create(session)
            val instance = ParametrizedTruncatedNormal(MEAN, STD_DEV, P1, P2, 12L)
            val operand = instance.initialize(FAN_IN, FAN_OUT, tf, shapeOperand(tf, shape), "default_name")
            operand.asOutput().tensor().copyTo(actual)

            assertArrayEquals(
                expected[0],
                actual[0],
                EPS
            )

            assertArrayEquals(
                expected[1],
                actual[1],
                EPS
            )

            assertEquals(
                "ParametrizedTruncatedNormal(mean=0.0, stdev=3.0, p1=-4.0, p2=4.0, seed=12)",
                instance.toString()
            )
        }
    }
}