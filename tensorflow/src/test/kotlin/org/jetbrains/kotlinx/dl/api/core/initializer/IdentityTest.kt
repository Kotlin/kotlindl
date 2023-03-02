/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.initializer

import org.jetbrains.kotlinx.dl.api.core.exception.IdentityDimensionalityException
import org.jetbrains.kotlinx.dl.api.core.shape.shapeOperand
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import org.tensorflow.EagerSession
import org.tensorflow.Shape
import org.tensorflow.op.Ops

private const val EPS = 1e-7f
private const val FAN_IN = 10
private const val FAN_OUT = 20

class IdentityTest {
    @Test
    fun initialize() {
        val actual = Array(2) { FloatArray(2) { 0f } }
        val expected = Array(2) { FloatArray(2) { 0f } }
        expected[0][0] = 1f
        expected[1][1] = 1f

        val shape = Shape.make(2, 2)

        EagerSession.create().use { session ->
            val tf = Ops.create(session)
            val instance = Identity()
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
                "Identity(scale=1.0)",
                instance.toString()
            )
        }
    }

    @Test
    fun initializeScaled() {
        val actual = Array(2) { FloatArray(2) { 0f } }
        val expected = Array(2) { FloatArray(2) { 0f } }
        expected[0][0] = 3.4f
        expected[1][1] = 3.4f

        val shape = Shape.make(2, 2)

        EagerSession.create().use { session ->
            val tf = Ops.create(session)
            val instance = Identity(3.4f)
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
                "Identity(scale=3.4)",
                instance.toString()
            )
        }
    }

    @Test
    fun initializeWithNonSquare2x3() {
        val actual = Array(2) { FloatArray(3) { 0f } }
        val expected = Array(3) { FloatArray(3) { 0f } }
        expected[0][0] = 1f
        expected[1][1] = 1f

        val shape = Shape.make(2, 3)

        EagerSession.create().use { session ->
            val tf = Ops.create(session)
            val instance = Identity()
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
                "Identity(scale=1.0)",
                instance.toString()
            )
        }
    }

    @Test
    fun initializeWithNonSquare3x2() {
        val actual = Array(3) { FloatArray(2) { 0f } }
        val expected = Array(3) { FloatArray(2) { 0f } }
        expected[0][0] = 1f
        expected[1][1] = 1f

        val shape = Shape.make(3, 2)

        EagerSession.create().use { session ->
            val tf = Ops.create(session)
            val instance = Identity()
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

            assertArrayEquals(
                expected[2],
                actual[2],
                EPS
            )

            assertEquals(
                "Identity(scale=1.0)",
                instance.toString()
            )
        }
    }

    @Test
    fun initializeWith1DShapeFails() {
        EagerSession.create().use { session ->
            val tf = Ops.create(session)
            val shape = Shape.make(5)

            val exception = assertThrows(IdentityDimensionalityException::class.java) {
                Identity().initialize(FAN_IN, FAN_OUT, tf, shapeOperand(tf, shape), "default_name")
            }

            assertEquals(
                "Identity matrix is not defined for order 1 tensors.",
                exception.message
            )
        }
    }

    @Test
    fun initializeWith3DShapeFails() {
        EagerSession.create().use { session ->
            val tf = Ops.create(session)
            val shape = Shape.make(5, 5, 5)

            val exception = assertThrows(IdentityDimensionalityException::class.java) {
                Identity().initialize(FAN_IN, FAN_OUT, tf, shapeOperand(tf, shape), "default_name")
            }

            assertEquals(
                "Identity matrix is not defined for order 3 tensors.",
                exception.message
            )
        }
    }
}
