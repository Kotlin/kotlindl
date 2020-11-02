/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.integration

import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.inference.savedmodel.Input
import org.jetbrains.kotlinx.dl.api.inference.savedmodel.Output
import org.jetbrains.kotlinx.dl.api.inference.savedmodel.SavedModel
import org.jetbrains.kotlinx.dl.datasets.Dataset
import org.jetbrains.kotlinx.dl.datasets.handlers.*
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class SavedModelTest {
    @Test
    fun basicInferenceOnMnist() {
        val PATH_TO_MODEL = "api/src/main/resources/models/savedmodel"

        fun main() {
            val (train, test) = Dataset.createTrainAndTestDatasets(
                TRAIN_IMAGES_ARCHIVE,
                TRAIN_LABELS_ARCHIVE,
                TEST_IMAGES_ARCHIVE,
                TEST_LABELS_ARCHIVE,
                NUMBER_OF_CLASSES,
                ::extractImages,
                ::extractLabels
            )

            SavedModel.load(PATH_TO_MODEL).use {
                println(it)

                it.reshape(::reshapeInput)
                it.input(Input.PLACEHOLDER)
                it.output(Output.ARGMAX)

                val prediction = it.predict(train.getX(0))

                assertEquals(train.getLabel(0), prediction)

                val predictions = it.predictAll(test)

                assertEquals(10000, predictions.size)
                assertTrue(it.evaluate(test, Metrics.ACCURACY) > 0.9)
            }
        }
    }

    fun reshapeInput(inputData: FloatArray): Array<Array<FloatArray>> {
        val reshaped = Array(
            1
        ) { Array(28) { FloatArray(28) } }
        for (i in inputData.indices) reshaped[0][i / 28][i % 28] = inputData[i]
        return reshaped
    }
}


