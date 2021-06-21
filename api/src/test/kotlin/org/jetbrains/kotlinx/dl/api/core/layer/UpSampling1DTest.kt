/*
 * Copyright 2021 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.UpSampling1D
import org.junit.jupiter.api.Test

internal class UpSampling1DTest {
    private val input = arrayOf(
        arrayOf(
            floatArrayOf(0.0f, 1.0f),
            floatArrayOf(2.0f, 3.0f),
            floatArrayOf(4.0f, 5.0f)
        )
    )

    @Test
    fun default() {
        val layer = UpSampling1D()
        val expected = arrayOf(
            arrayOf(
                floatArrayOf(0.0f, 1.0f),
                floatArrayOf(0.0f, 1.0f),
                floatArrayOf(2.0f, 3.0f),
                floatArrayOf(2.0f, 3.0f),
                floatArrayOf(4.0f, 5.0f),
                floatArrayOf(4.0f, 5.0f)
            )
        )
        TODO("Implement test case")
    }

    @Test
    fun testWithNonDefaultSize() {
        val layer = UpSampling1D(size = 3)
        val expected = arrayOf(
            arrayOf(
                floatArrayOf(0.0f, 1.0f),
                floatArrayOf(0.0f, 1.0f),
                floatArrayOf(0.0f, 1.0f),
                floatArrayOf(2.0f, 3.0f),
                floatArrayOf(2.0f, 3.0f),
                floatArrayOf(2.0f, 3.0f),
                floatArrayOf(4.0f, 5.0f),
                floatArrayOf(4.0f, 5.0f),
                floatArrayOf(4.0f, 5.0f)
            )
        )
        TODO("Implement test case")
    }
}
