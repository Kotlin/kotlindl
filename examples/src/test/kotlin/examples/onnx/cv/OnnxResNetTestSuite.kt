/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.cv

import examples.onnx.cv.resnet.notop.resnet50CustomPrediction
import examples.onnx.cv.resnet.notop.resnet50additionalTraining
import examples.onnx.cv.efficicentnet.efficientNet4LitePrediction
import examples.onnx.cv.resnet.*
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.junit.jupiter.api.Test

class OnnxResNetTestSuite {
    @Test
    fun resnet18predictionTest() {
        resnet18prediction()
    }

    @Test
    fun resnet18easyPredictionTest() {
        resnet18LightAPIPrediction()
    }

    @Test
    fun resnet18v2predictionTest() {
        runImageRecognitionPrediction(ONNXModels.CV.ResNet18v2())
    }

    @Test
    fun resnet34predictionTest() {
        runImageRecognitionPrediction(ONNXModels.CV.ResNet34())
    }

    @Test
    fun resnet34v2predictionTest() {
        runImageRecognitionPrediction(ONNXModels.CV.ResNet34v2())
    }

    @Test
    fun resnet50predictionTest() {
        runImageRecognitionPrediction(ONNXModels.CV.ResNet50())
    }

    @Test
    fun resnet50v2predictionTest() {
        runImageRecognitionPrediction(ONNXModels.CV.ResNet50v2())
    }

    @Test
    fun resnet50additionalTrainingTest() {
        resnet50additionalTraining()
    }

    @Test
    fun resnet50CustomPredictionTest() {
        resnet50CustomPrediction()
    }

    @Test
    fun resnet101predictionTest() {
        runImageRecognitionPrediction(ONNXModels.CV.ResNet101())
    }

    @Test
    fun resnet101v2predictionTest() {
        runImageRecognitionPrediction(ONNXModels.CV.ResNet101v2())
    }

    @Test
    fun resnet152predictionTest() {
        runImageRecognitionPrediction(ONNXModels.CV.ResNet152())
    }

    @Test
    fun resnet152v2predictionTest() {
        runImageRecognitionPrediction(ONNXModels.CV.ResNet152v2())
    }

    @Test
    fun efficientNet4LitePredictionTest(){
        efficientNet4LitePrediction()
    }
}
