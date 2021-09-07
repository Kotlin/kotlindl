/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.modelzoo.resnet

import examples.transferlearning.modelzoo.resnet.resnet50.resnet50additionalTraining
import examples.transferlearning.modelzoo.resnet.resnet50.resnet50copyModelPrediction
import examples.transferlearning.modelzoo.resnet.resnet50.resnet50prediction
import org.junit.jupiter.api.Test

class ResNetTestSuite {
    @Test
    fun resnet18predictionTest() {
        resnet18prediction()
    }

    @Test
    fun resnet34predictionTest() {
        resnet34prediction()
    }

    @Test
    fun resnet50predictionTest() {
        resnet50prediction()
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
        resnet50v2prediction()
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
}
