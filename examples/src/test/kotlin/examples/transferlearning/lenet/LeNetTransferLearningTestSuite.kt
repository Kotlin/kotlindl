/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.lenet

import org.junit.jupiter.api.Test

class LeNetTransferLearningTestSuite {
    @Test
    fun loadModelWithWeightsAndEvaluateTest() {
        loadModelWithWeightsAndEvaluate()
    }

    @Test
    fun loadModelWithoutWeightsInitAndEvaluateTest() {
        loadModelWithoutWeightsInitAndEvaluate()
    }

    @Test
    fun additionalTrainingTest() {
        additionalTraining()
    }

    @Test
    fun additionalTrainingAndFreezingTest() {
        additionalTrainingAndFreezing()
    }

    @Test
    fun additionalTrainingAndPartialFreezingAndPartialInitializationTest() {
        additionalTrainingAndPartialFreezingAndPartialInitialization()
    }

    @Test
    fun additionalTrainingAndNewTopDenseLayersTest() {
        additionalTrainingAndNewTopDenseLayers()
    }
}
