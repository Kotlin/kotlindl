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

internal const val EPS: Float = 1e-2f

internal class PoissonTest {
    @Test
    fun zeroTest() {
        val yTrueArray = floatArrayOf(0f, 2f, 3f, 4f, 5f, 6f)

        EagerSession.create().use { session ->
            val tf = Ops.create(session)
            val instance = Poisson()

            val yTrue: Operand<Float> = tf.reshape(tf.constant(yTrueArray), tf.constant(intArrayOf(2, 3)))

            val numberOfLosses = tf.constant(yTrue.asOutput().shape().numElements().toFloat())

            assertEquals(6f, numberOfLosses.asOutput().tensor().floatValue())

            val operand: Operand<Float> = instance.apply(tf, yTrue, yTrue, numberOfLosses)

            assertEquals(
                -1.5041842f,
                operand.asOutput().tensor().floatValue(),
                EPS
            )
        }
    }

    @Test
    fun simple() {
        val yTrueArray = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)
        val yPredArray = floatArrayOf(0f, 2f, 3f, 4f, 5f, 6f)

        EagerSession.create().use { session ->
            val tf = Ops.create(session)
            val instance = Poisson()

            val yTrue: Operand<Float> = tf.reshape(tf.constant(yTrueArray), tf.constant(intArrayOf(2, 3)))
            val yPred: Operand<Float> = tf.reshape(tf.constant(yPredArray), tf.constant(intArrayOf(2, 3)))

            val numberOfLosses = tf.constant(yPred.asOutput().shape().numElements().toFloat())

            assertEquals(6f, numberOfLosses.asOutput().tensor().floatValue())

            val operand: Operand<Float> = instance.apply(tf, yPred = yPred, yTrue = yTrue, numberOfLosses)

            assertEquals(
                0.41463676f,
                operand.asOutput().tensor().floatValue(),
                EPS
            )
        }
    }

    @Test
    fun basic() {
        val yTrueArray = floatArrayOf(4f, 8f, 12f, 8f, 1f, 3f)
        val yPredArray = floatArrayOf(1f, 9f, 2f, 5f, 2f, 6f)

        EagerSession.create().use { session ->
            val tf = Ops.create(session)
            val instance = Poisson(reductionType = ReductionType.SUM_OVER_BATCH_SIZE)

            val yTrue: Operand<Float> = tf.reshape(tf.constant(yTrueArray), tf.constant(intArrayOf(2, 3)))
            val yPred: Operand<Float> = tf.reshape(tf.constant(yPredArray), tf.constant(intArrayOf(2, 3)))

            val numberOfLosses = tf.constant(yPred.asOutput().shape().numElements().toFloat())

            assertEquals(6f, numberOfLosses.asOutput().tensor().floatValue())

            val operand: Operand<Float> = instance.apply(tf, yPred = yPred, yTrue = yTrue, numberOfLosses)

            assertEquals(
                -3.3066044f,
                operand.asOutput().tensor().floatValue(),
                EPS
            )
        }
    }

    @Test
    fun sumReduction() {
        val yTrueArray = floatArrayOf(4f, 8f, 12f, 8f, 1f, 3f)
        val yPredArray = floatArrayOf(1f, 9f, 2f, 5f, 2f, 6f)

        EagerSession.create().use { session ->
            val tf = Ops.create(session)
            val instance = Poisson(reductionType = ReductionType.SUM)

            val yTrue: Operand<Float> = tf.reshape(tf.constant(yTrueArray), tf.constant(intArrayOf(2, 3)))
            val yPred: Operand<Float> = tf.reshape(tf.constant(yPredArray), tf.constant(intArrayOf(2, 3)))

            val operand: Operand<Float> = instance.apply(tf, yPred = yPred, yTrue = yTrue, null)

            assertEquals(
                -6.613209f,
                operand.asOutput().tensor().floatValue(),
                EPS
            )
        }
    }
}
