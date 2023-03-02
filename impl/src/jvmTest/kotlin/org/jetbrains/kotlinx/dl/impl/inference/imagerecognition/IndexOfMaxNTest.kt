/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.inference.imagerecognition

import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test

class IndexOfMaxNTest {
    @Test
    fun testIndexOfMaxN() {
        Assertions.assertEquals(listOf(3, 0), floatArrayOf(0.1f, 0f, -10f, 20f).indexOfMaxN(2))
        Assertions.assertEquals(listOf(3, 2, 1), floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f).indexOfMaxN(3))
        Assertions.assertEquals(listOf(3, 4), floatArrayOf(0.239f, -239f, -2.39f, 239f, 23.9f).indexOfMaxN(2))
    }
}