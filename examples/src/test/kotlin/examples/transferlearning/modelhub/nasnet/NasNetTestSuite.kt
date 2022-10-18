/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.modelhub.nasnet

import examples.transferlearning.runImageRecognitionTransferLearning
import org.jetbrains.kotlinx.dl.api.inference.loaders.TFModels
import org.junit.jupiter.api.Test

class NasNetTestSuite {
    @Test
    fun nasNetMobilePredictionTest() {
        nasNetMobilePrediction()
    }

    @Test
    fun nasNetMobileAdditionalTrainingNoTopTest() {
        runImageRecognitionTransferLearning(modelType = TFModels.CV.NASNetMobile(noTop = true))
    }

    @Test
    fun nasNetLargePredictionTest() {
        nasNetLargePrediction()
    }

    @Test
    fun nasNetLargeAdditionalTrainingNoTopTest() {
        runImageRecognitionTransferLearning(
            modelType = TFModels.CV.NASNetLarge(
                noTop = true,
                inputShape = intArrayOf(370, 370, 3)
            )
        )
    }
}
