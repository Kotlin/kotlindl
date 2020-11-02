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
        EagerSession.create().use { session ->
            val tf = Ops.create(session)

            val operand = instance.apply(tf, tf.constant(input))
            operand.asOutput().tensor().copyTo(actual)

            Assertions.assertArrayEquals(
                expected,
                actual,
                EPS
            )
        }
    }
}