/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.modelhub.resnet

import examples.transferlearning.runImageRecognitionPrediction
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.TFModels
import org.junit.jupiter.api.Test

class ResNetTestSuite {
    @Test
    fun resnet18predictionTest() {
        runImageRecognitionPrediction(modelType = TFModels.CV.ResNet18)
    }

    @Test
    fun resnet34predictionTest() {
        runImageRecognitionPrediction(modelType = TFModels.CV.ResNet34)
    }

    @Test
    fun resnet50predictionTest() {
        resnet50easyPrediction()
    }

    @Test
    fun resnet50additionalTrainingTest() {
        resnet50additionalTraining()
    }

    @Test
    fun resnet50copyModelPredictionTest() {
        resnet50copyModelPrediction()
    }

    @Test
    fun resnet50v2predictionTest() {
        runImageRecognitionPrediction(modelType = TFModels.CV.ResNet50v2)
    }

    @Test
    fun resnet101predictionTest() {
        runImageRecognitionPrediction(modelType = TFModels.CV.ResNet101)
    }

    @Test
    fun resnet101v2predictionTest() {
        runImageRecognitionPrediction(modelType = TFModels.CV.ResNet101v2)
    }

    @Test
    fun resnet152predictionTest() {
        runImageRecognitionPrediction(modelType = TFModels.CV.ResNet152)
    }

    @Test
    fun resnet152v2predictionTest() {
        runImageRecognitionPrediction(modelType = TFModels.CV.ResNet152v2)
    }
}
