/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.cnn.mnist

import examples.cnn.mnist.advanced.lenetWithAlternativeLossFunction
import examples.cnn.mnist.advanced.lenetWithEarlyStoppingCallback
import examples.cnn.mnist.advanced.modernLenetWithRegularizers
import org.junit.jupiter.api.Test

class MnistTestSuite {
    @Test
    fun denseOnlyTest() {
        denseOnly()
    }

    @Test
    fun lenetClassicTest() {
        lenetClassic()
    }

    @Test
    fun lenetWithAlternativeLossFunctionTest() {
        lenetWithAlternativeLossFunction()
    }

    @Test
    fun lenetWithEarlyStoppingCallbackTest() {
        lenetWithEarlyStoppingCallback()
    }

    @Test
    fun modernLenetTest() {
        modernLenet()
    }

    @Test
    fun modernLenetWithRegularizersTest() {
        modernLenetWithRegularizers()
    }

    @Test
    fun vggTest() {
        vgg()
    }
}
