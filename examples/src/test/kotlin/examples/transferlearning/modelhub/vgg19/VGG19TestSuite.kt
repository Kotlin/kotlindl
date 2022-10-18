/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.modelhub.vgg19

import org.junit.jupiter.api.Test

class VGG19TestSuite {
    @Test
    fun vgg19predictionTest() {
        vgg19prediction()
    }

    @Test
    fun vgg19additionalTrainingTest() {
        vgg19additionalTraining()
    }

    @Test
    fun vgg19copyModelPredictionTest() {
        vgg19copyModelPrediction()
    }
}
