/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.modelhub.densenet

import org.junit.jupiter.api.Test

class DenseNetTestSuite {
    @Test
    fun denseNet121PredictionTest() {
        denseNet121Prediction()
    }

    @Test
    fun denseNet169PredictionTest() {
        denseNet169Prediction()
    }

    @Test
    fun denseNet201PredictionTest() {
        denseNet201Prediction()
    }
}
