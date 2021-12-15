/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.modelhub.densenet

import examples.transferlearning.runImageRecognitionTransferLearningOnTopModel
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.TFModels
import org.junit.jupiter.api.Test

class DenseNetTestSuite {
    @Test
    fun denseNet121PredictionTest() {
        denseNet121Prediction()
    }

    @Test
    fun denseNet121AdditionalTrainingTest() {
        runImageRecognitionTransferLearningOnTopModel(
            modelType = TFModels.CV.DenseNet121(
                inputShape = intArrayOf(80, 80, 3)
            ),
            epochs = 2
        )
    }

    @Test
    fun denseNet169PredictionTest() {
        denseNet169Prediction()
    }

    @Test
    fun denseNet169AdditionalTrainingTest() {
        runImageRecognitionTransferLearningOnTopModel(
            modelType = TFModels.CV.DenseNet169(
                inputShape = intArrayOf(80, 80, 3)
            )
        )
    }


    @Test
    fun denseNet201PredictionTest() {
        denseNet201Prediction()
    }

    @Test
    fun denseNet201AdditionalTrainingTest() {
        runImageRecognitionTransferLearningOnTopModel(
            modelType = TFModels.CV.DenseNet201(
                inputShape = intArrayOf(110, 110, 3)
            )
        )
    }
}
