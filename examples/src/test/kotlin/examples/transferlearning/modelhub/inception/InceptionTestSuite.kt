/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.modelhub.inception

import org.junit.jupiter.api.Test

class InceptionTestSuite {
    @Test
    fun inceptionV3PredictionTest() {
        inceptionV3Prediction()
    }

    @Test
    fun xceptionPredictionTest() {
        xceptionPrediction()
    }
}
