/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.cv

import examples.onnx.cv.efficicentnet.efficientNetB0Prediction
import examples.onnx.cv.efficicentnet.lightAPI.efficientNetB0EasyPrediction
import examples.onnx.cv.efficicentnet.lightAPI.efficientNetB7LightAPIPrediction
import examples.onnx.cv.efficicentnet.notop.efficientNetB0AdditionalTraining
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModels
import org.junit.jupiter.api.Test

class OnnxEfficientNetTestSuite {
    @Test
    fun efficientNetB0PredictionTest() {
        efficientNetB0Prediction()
    }

    @Test
    fun efficientNetB0AdditionalTrainingTest() {
        efficientNetB0AdditionalTraining()
    }

    @Test
    fun efficientNetB0EasyPredictionTest() {
        efficientNetB0EasyPrediction()
    }

    @Test
    fun efficientNetB1PredictionTest() {
        runONNXImageRecognitionPrediction(ONNXModels.CV.EfficientNetB1())
    }

    @Test
    fun efficientNetB1AdditionalTrainingTest() {
        runONNXAdditionalTraining(ONNXModels.CV.EfficientNetB1(noTop = true))
    }

    @Test
    fun efficientNetB2PredictionTest() {
        runONNXImageRecognitionPrediction(ONNXModels.CV.EfficientNetB2())
    }

    @Test
    fun efficientNetB2AdditionalTrainingTest() {
        runONNXAdditionalTraining(ONNXModels.CV.EfficientNetB2(noTop = true))
    }

    @Test
    fun efficientNetB3PredictionTest() {
        runONNXImageRecognitionPrediction(ONNXModels.CV.EfficientNetB3())
    }

    @Test
    fun efficientNetB3AdditionalTrainingTest() {
        runONNXAdditionalTraining(ONNXModels.CV.EfficientNetB3(noTop = true))
    }

    @Test
    fun efficientNetB4PredictionTest() {
        runONNXImageRecognitionPrediction(ONNXModels.CV.EfficientNetB4())
    }

    @Test
    fun efficientNetB4AdditionalTrainingTest() {
        runONNXAdditionalTraining(ONNXModels.CV.EfficientNetB4(noTop = true))
    }

    @Test
    fun efficientNetB5PredictionTest() {
        runONNXImageRecognitionPrediction(ONNXModels.CV.EfficientNetB5())
    }

    @Test
    fun efficientNetB5AdditionalTrainingTest() {
        runONNXAdditionalTraining(ONNXModels.CV.EfficientNetB5(noTop = true))
    }

    @Test
    fun efficientNetB6PredictionTest() {
        runONNXImageRecognitionPrediction(ONNXModels.CV.EfficientNetB6())
    }

    @Test
    fun efficientNetB6AdditionalTrainingTest() {
        runONNXAdditionalTraining(ONNXModels.CV.EfficientNetB6(noTop = true))
    }

    @Test
    fun efficientNetB7PredictionTest() {
        runONNXImageRecognitionPrediction(ONNXModels.CV.EfficientNetB7())
    }

    @Test
    fun efficientNetB7AdditionalTrainingTest() {
        runONNXAdditionalTraining(ONNXModels.CV.EfficientNetB7(noTop = true))
    }

    @Test
    fun efficientNetB7EasyPredictionTest() {
        efficientNetB7LightAPIPrediction()
    }
}


