/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.activation

import org.junit.jupiter.api.Assertions
import org.tensorflow.EagerSession
import org.tensorflow.op.Ops

const val EPS: Float = 1e-2f

open class ActivationTest {
    protected fun assertActivationFunction(
        instance: Activation,
        input: FloatArray,
        actual: FloatArray,
        expected: FloatArray
    ) {
        assertActivationFunction(instance, input, expected)
    }

    protected fun assertActivationFunction(
        act: Activation,
        inp: FloatArray,
        exp: FloatArray
    ) {
        EagerSession.create().use { session ->
            val tf = Ops.create(session)
            Assertions.assertArrayEquals(exp, act.apply(tf, tf.constant(inp)).asOutput().tensor().copyTo(inp), EPS)
        }
    }

    protected fun assertActivationFunction(
        act: Activation,
        inp: Array<FloatArray>,
        exp: Array<FloatArray>
    ) {
        /**
         * Accepts 2D float input values
         */

        // Higher Dimensional assertArrayEquals have no delta option
        // Due to numeric instability we loop and use 1D version with EPS value
        EagerSession.create().use { session ->
            val tf = Ops.create(session)
            val actual = act.apply(tf, tf.constant(inp)).asOutput().tensor().copyTo(inp)
            for (i in 0..exp.lastIndex) {
                Assertions.assertArrayEquals(exp[i], actual[i], EPS)
            }
        }
    }

    protected fun assertActivationFunction(
        act: Activation,
        inp: Array<Array<FloatArray>>,
        exp: Array<Array<FloatArray>>
    ) {
        /**
         * Accepts 3D float input values
         */
        EagerSession.create().use { session ->
            val tf = Ops.create(session)
            val actual = act.apply(tf, tf.constant(inp)).asOutput().tensor().copyTo(inp)
            for (i in 0..exp.lastIndex) {
                for (j in 0..exp[i].lastIndex) {
                    Assertions.assertArrayEquals(exp[i][j], actual[i][j], EPS)
                }
            }
        }
    }
}
