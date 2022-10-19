/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.modelhub.mobilenet

import examples.transferlearning.runImageRecognitionTransferLearning
import org.jetbrains.kotlinx.dl.api.inference.loaders.TFModels
import org.junit.jupiter.api.Test

class MobileNetTestSuite {
    @Test
    fun mobilenetPredictionTest() {
        mobileNetPrediction()
    }

    @Test
    fun mobilenetWithAdditionalTrainingTest() {
        mobilenetWithAdditionalTraining()
    }

    @Test
    fun mobilenetAdditionalTrainingNoTopTest() {
        runImageRecognitionTransferLearning(
            modelType = TFModels.CV.MobileNet(
                noTop = true,
                inputShape = intArrayOf(100, 100, 3)
            )
        )
    }

    @Test
    fun mobilenetv2PredictionTest() {
        mobileNetV2Prediction()
    }

    @Test
    fun mobilenetv2AdditionalTrainingNoTopTest() {
        runImageRecognitionTransferLearning(
            modelType = TFModels.CV.MobileNetV2(
                noTop = true,
                inputShape = intArrayOf(120, 120, 3)
            )
        )
    }
}
