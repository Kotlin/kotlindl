/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.loss

import org.jetbrains.kotlinx.dl.api.core.shape.numElements
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import org.tensorflow.EagerSession
import org.tensorflow.Operand
import org.tensorflow.op.Ops

internal class HuberTest {
    @Test
    fun zeroTest() {
        val yTrueArray = floatArrayOf(0f, 2f, 3f, 4f, 5f, 6f)

        EagerSession.create().use { session ->
            val tf = Ops.create(session)
            val instance = Huber()

            val yTrue: Operand<Float> = tf.reshape(tf.constant(yTrueArray), tf.constant(intArrayOf(2, 3)))

            val numberOfLosses = tf.constant(yTrue.asOutput().shape().numElements().toFloat())

            assertEquals(6f, numberOfLosses.asOutput().tensor().floatValue())

            val operand: Operand<Float> = instance.apply(tf, yTrue, yTrue, numberOfLosses)

            assertEquals(
                0.0f,
                operand.asOutput().tensor().floatValue(),
                EPS
            )
        }
    }

    @Test
    fun simple() {
        val yPredArray = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)
        val yTrueArray = floatArrayOf(0f, 2f, 3f, 4f, 5f, 6f)

        EagerSession.create().use { session ->
            val tf = Ops.create(session)
            val instance = Huber()

            val yTrue: Operand<Float> = tf.reshape(tf.constant(yTrueArray), tf.constant(intArrayOf(2, 3)))
            val yPred: Operand<Float> = tf.reshape(tf.constant(yPredArray), tf.constant(intArrayOf(2, 3)))

            val numberOfLosses = tf.constant(yPred.asOutput().shape().numElements().toFloat())

            assertEquals(6f, numberOfLosses.asOutput().tensor().floatValue())

            val operand: Operand<Float> = instance.apply(tf, yPred = yPred, yTrue = yTrue, numberOfLosses)

            assertEquals(
                0.083333336f,
                operand.asOutput().tensor().floatValue(),
                EPS
            )
        }
    }

    @Test
    fun basic() {
        val yTrueArray = floatArrayOf(.9f, .2f, .2f, .8f, .4f, .6f)
        val yPredArray = floatArrayOf(1f, 0f, 1f, 1f, 0f, 0f)

        EagerSession.create().use { session ->
            val tf = Ops.create(session)
            val instance = Huber(reductionType = ReductionType.SUM_OVER_BATCH_SIZE)

            val yTrue: Operand<Float> = tf.reshape(tf.constant(yTrueArray), tf.constant(intArrayOf(2, 3)))
            val yPred: Operand<Float> = tf.reshape(tf.constant(yPredArray), tf.constant(intArrayOf(2, 3)))

            val numberOfLosses = tf.constant(yPred.asOutput().shape().numElements().toFloat())

            assertEquals(6f, numberOfLosses.asOutput().tensor().floatValue())

            val operand: Operand<Float> = instance.apply(tf, yPred = yPred, yTrue = yTrue, numberOfLosses)

            assertEquals(
                0.10416668f,
                operand.asOutput().tensor().floatValue(),
                EPS
            )
        }
    }

    @Test
    fun sumReduction() {
        val yTrueArray = floatArrayOf(.9f, .2f, .2f, .8f, .4f, .6f)
        val yPredArray = floatArrayOf(1f, 0f, 1f, 1f, 0f, 0f)

        EagerSession.create().use { session ->
            val tf = Ops.create(session)
            val instance = Huber(reductionType = ReductionType.SUM)

            val yTrue: Operand<Float> = tf.reshape(tf.constant(yTrueArray), tf.constant(intArrayOf(2, 3)))
            val yPred: Operand<Float> = tf.reshape(tf.constant(yPredArray), tf.constant(intArrayOf(2, 3)))

            val operand: Operand<Float> = instance.apply(tf, yPred = yPred, yTrue = yTrue, null)

            assertEquals(
                0.20833334f,
                operand.asOutput().tensor().floatValue(),
                EPS
            )
        }
    }
}
