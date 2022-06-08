/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.modelhub.inception

import examples.transferlearning.runImageRecognitionTransferLearning
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.TFModels
import org.junit.jupiter.api.Test

class InceptionTestSuite {
    @Test
    fun inceptionV3PredictionTest() {
        inceptionV3Prediction()
    }

    @Test
    fun inceptionV3dditionalTrainingNoTopTest() {
        runImageRecognitionTransferLearning(
            modelType = TFModels.CV.Inception(
                noTop = true,
                inputShape = intArrayOf(99, 99, 3)
            )
        )
    }

    @Test
    fun xceptionPredictionTest() {
        xceptionPrediction()
    }

    @Test
    fun xceptionAdditionalTrainingNoTopTest() {
        runImageRecognitionTransferLearning(
            modelType = TFModels.CV.Xception(
                noTop = true,
                inputShape = intArrayOf(90, 90, 3)
            )
        )
    }
}
