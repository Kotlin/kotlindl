/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx

import examples.onnx.cv.custom.resnet50CustomPrediction
import examples.onnx.cv.custom.resnet50additionalTraining
import examples.onnx.cv.efficicentnet.efficientNet4LitePrediction
import examples.onnx.cv.resnet.*
import org.junit.jupiter.api.Test

class OnnxResNetTestSuite {
    @Test
    fun resnet18predictionTest() {
        resnet18prediction()
    }

    @Test
    fun resnet18v2predictionTest() {
        resnet18v2prediction()
    }

    @Test
    fun resnet34predictionTest() {
        resnet34prediction()
    }

    @Test
    fun resnet34v2predictionTest() {
        resnet34v2prediction()
    }

    @Test
    fun resnet50predictionTest() {
        resnet50prediction()
    }

    @Test
    fun resnet50v2predictionTest() {
        resnet50v2prediction()
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
        resnet101prediction()
    }

    @Test
    fun resnet101v2predictionTest() {
        resnet101v2prediction()
    }

    @Test
    fun resnet152predictionTest() {
        resnet152prediction()
    }

    @Test
    fun resnet152v2predictionTest() {
        resnet152v2prediction()
    }

    @Test
    fun efficientNet4LitePredictionTest(){
        efficientNet4LitePrediction()
    }
}
