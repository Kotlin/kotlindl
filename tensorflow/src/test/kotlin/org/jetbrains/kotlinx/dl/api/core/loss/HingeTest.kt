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

internal class HingeTest {
    @Test
    fun zeroTest() {
        val yTrueArray = floatArrayOf(0f, 2f, 3f, 4f, 5f, 6f)

        EagerSession.create().use { session ->
            val tf = Ops.create(session)
            val instance = Hinge()

            val yTrue: Operand<Float> = tf.reshape(tf.constant(yTrueArray), tf.constant(intArrayOf(2, 3)))

            val numberOfLosses = tf.constant(yTrue.asOutput().shape().numElements().toFloat())

            assertEquals(6f, numberOfLosses.asOutput().tensor().floatValue())

            val operand: Operand<Float> = instance.apply(tf, yTrue, yTrue, numberOfLosses)

            assertEquals(
                0.16666667f,
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
            val instance = Hinge()

            val yTrue: Operand<Float> = tf.reshape(tf.constant(yTrueArray), tf.constant(intArrayOf(2, 3)))
            val yPred: Operand<Float> = tf.reshape(tf.constant(yPredArray), tf.constant(intArrayOf(2, 3)))

            val numberOfLosses = tf.constant(yPred.asOutput().shape().numElements().toFloat())

            assertEquals(6f, numberOfLosses.asOutput().tensor().floatValue())

            val operand: Operand<Float> = instance.apply(tf, yPred = yPred, yTrue = yTrue, numberOfLosses)

            assertEquals(
                0.33333334f,
                operand.asOutput().tensor().floatValue(),
                EPS
            )
        }
    }

    @Test
    fun basic() {
        val yTrueArray = floatArrayOf(0f, 1f, 0f, 1f, 0f, 0f, 1f, 1f)
        val yPredArray = floatArrayOf(-0.3f, 0.2f, -0.1f, 1.6f, -0.25f, -1f, 0.5f, 0.6f)

        EagerSession.create().use { session ->
            val tf = Ops.create(session)
            val instance = Hinge(reductionType = ReductionType.SUM_OVER_BATCH_SIZE)

            val yTrue: Operand<Float> = tf.reshape(tf.constant(yTrueArray), tf.constant(intArrayOf(2, 4)))
            val yPred: Operand<Float> = tf.reshape(tf.constant(yPredArray), tf.constant(intArrayOf(2, 4)))

            val numberOfLosses = tf.constant(yPred.asOutput().shape().numElements().toFloat())

            assertEquals(8f, numberOfLosses.asOutput().tensor().floatValue())

            val operand: Operand<Float> = instance.apply(tf, yPred = yPred, yTrue = yTrue, numberOfLosses)

            assertEquals(
                0.50625f,
                operand.asOutput().tensor().floatValue(),
                EPS
            )
        }
    }

    @Test
    fun sumReduction() {
        val yTrueArray = floatArrayOf(0f, 1f, 0f, 1f, 0f, 0f, 1f, 1f)
        val yPredArray = floatArrayOf(-0.3f, 0.2f, -0.1f, 1.6f, -0.25f, -1f, 0.5f, 0.6f)

        EagerSession.create().use { session ->
            val tf = Ops.create(session)
            val instance = Hinge(reductionType = ReductionType.SUM)

            val yTrue: Operand<Float> = tf.reshape(tf.constant(yTrueArray), tf.constant(intArrayOf(2, 4)))
            val yPred: Operand<Float> = tf.reshape(tf.constant(yPredArray), tf.constant(intArrayOf(2, 4)))

            val operand: Operand<Float> = instance.apply(tf, yPred = yPred, yTrue = yTrue, null)

            assertEquals(
                1.0125f,
                operand.asOutput().tensor().floatValue(),
                EPS
            )
        }
    }
}
