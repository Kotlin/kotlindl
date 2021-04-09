/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

import examples.cnn.mnist.denseOnly
import examples.cnn.mnist.lenetClassic
import org.junit.jupiter.api.Test

class CNNTest {
    @Test
    fun denseOnlyTest() {
        denseOnly()
    }

    @Test
    fun lenetClassicTest() {
        lenetClassic()
    }
}
