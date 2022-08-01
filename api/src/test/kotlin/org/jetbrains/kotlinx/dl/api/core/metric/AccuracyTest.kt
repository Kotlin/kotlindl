/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.metric

import org.jetbrains.kotlinx.dl.api.core.shape.numElements
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import org.tensorflow.EagerSession
import org.tensorflow.Operand
import org.tensorflow.op.Ops

internal class AccuracyTest {
    @Test
    fun zeroTest() {
        val yTrueArray = floatArrayOf(0f, 2f, 3f, 4f, 5f, 6f)

        EagerSession.create().use { session ->
            val tf = Ops.create(session)
            val instance = Accuracy()

            val yTrue: Operand<Float> = tf.reshape(tf.constant(yTrueArray), tf.constant(intArrayOf(2, 3)))

            val numberOfLosses = tf.constant(yTrue.asOutput().shape().numElements().toFloat())

            assertEquals(6f, numberOfLosses.asOutput().tensor().floatValue())

            val operand: Operand<Float> = instance.apply(tf, yTrue, yTrue, numberOfLosses)

            assertEquals(
                1.0f,
                operand.asOutput().tensor().floatValue()
            )
        }
    }

    @Test
    fun simpleTest() {
        val yPredArray = floatArrayOf(1f, 0f, 0f, 1f, 0f, 0f)
        val yTrueArray = floatArrayOf(0f, 1f, 0f, 1f, 0f, 0f)
        // 2 rows with 3 labels, different for the first row, same for the second row

        EagerSession.create().use { session ->
            val tf = Ops.create(session)
            val instance = Accuracy()

            val yTrue: Operand<Float> = tf.reshape(tf.constant(yTrueArray), tf.constant(intArrayOf(2, 3)))
            val yPred: Operand<Float> = tf.reshape(tf.constant(yPredArray), tf.constant(intArrayOf(2, 3)))

            val numberOfLosses = tf.constant(yPred.asOutput().shape().numElements().toFloat())

            assertEquals(6f, numberOfLosses.asOutput().tensor().floatValue())

            val operand: Operand<Float> = instance.apply(tf, yPred = yPred, yTrue = yTrue, numberOfLosses)

            assertEquals(
                0.5f,
                operand.asOutput().tensor().floatValue()
            )
        }
    }

    @Test
    fun simpleTest3() {
        val yTrueArray = floatArrayOf(1f, 9f, 2f, -5f, -2f, 6f)
        val yPredArray = floatArrayOf(4f, 8f, 12f, 8f, 1f, 3f)

        EagerSession.create().use { session ->
            val tf = Ops.create(session)
            val instance = Accuracy()

            val yTrue: Operand<Float> = tf.reshape(tf.constant(yTrueArray), tf.constant(intArrayOf(2, 3)))
            val yPred: Operand<Float> = tf.reshape(tf.constant(yPredArray), tf.constant(intArrayOf(2, 3)))

            val numberOfLosses = tf.constant(yPred.asOutput().shape().numElements().toFloat())

            assertEquals(6f, numberOfLosses.asOutput().tensor().floatValue())

            val operand: Operand<Float> = instance.apply(tf, yPred = yPred, yTrue = yTrue, numberOfLosses)

            assertEquals(
                0.0f,
                operand.asOutput().tensor().floatValue()
            )
        }
    }
}
