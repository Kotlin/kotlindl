/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.activation

import org.junit.jupiter.api.Assertions
import org.tensorflow.EagerSession
import org.tensorflow.op.Ops

const val EPS: Float = 1e-2f

open class ActivationTest {

    /**
     * Checks if an Activation-function object with input [inp] gives valid result of [exp].
     *
     * For example, if you want to test [ReluActivation] with [inp] and [exp]
     * ```
     *      val act = ReluActivation()
     *      val inp = floatArrayOf(-1f, 0f, 1f)
     *      val exp = floatArrayOf( 0f, 0f, 1f)
     *
     *      assertActivationFunction(act, inp, exp) // Test passes
     * ```
     *
     * @param act activation function object
     * @param inp FloatArray applied to the activation function
     * @param exp FloatArray expected output for an activation function object [act] with input [inp]
     */
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

    /**
     * Checks if an Activation-function object with input [inp] gives a valid result of [exp].
     *
     * For example, if you want to test [ReluActivation] with [inp] and [exp]
     * ```
     *      val act = ReluActivation()
     *      val inp = arrayOf(floatArrayOf(-1f, -1f, -1f),
     *                        floatArrayOf(0f, 0f, 0f),
     *                        floatArrayOf(1f, 1f, 1f)
     *                        )
     *      val exp = arrayOf(floatArrayOf( 0f, 0f, 0f),
     *                        floatArrayOf(0f, 0f, 0f),
     *                        floatArrayOf(1f, 1f, 1f)
     *                        )
     *
     *      assertActivationFunction(act, inp, exp) // Test passes
     * ```
     *
     * @param act activation function object
     * @param inp Array`<FloatArray`> applied to the activation function
     * @param exp Array`<FloatArray`> expected output for an activation function object [act] with input [inp]
     */
    protected fun assertActivationFunction(
        act: Activation,
        inp: Array<FloatArray>,
        exp: Array<FloatArray>
    ) {
        // Higher Dimensional assertArrayEquals have no delta option
        EagerSession.create().use { session ->
            val tf = Ops.create(session)
            val actual = act.apply(tf, tf.constant(inp)).asOutput().tensor().copyTo(inp)
            for (i in 0..exp.lastIndex) {
                Assertions.assertArrayEquals(exp[i], actual[i], EPS)
            }
        }
    }

    /**
     * Accepts 3D float input values
     *
     * @see assertActivationFunction
     */
    protected fun assertActivationFunction(
        act: Activation,
        inp: Array<Array<FloatArray>>,
        exp: Array<Array<FloatArray>>
    ) {
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
