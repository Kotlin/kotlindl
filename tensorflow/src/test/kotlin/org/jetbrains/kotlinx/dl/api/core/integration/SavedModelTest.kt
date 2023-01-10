/*
 * Copyright 2020-2023 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.integration

import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.inference.savedmodel.SavedModel
import org.jetbrains.kotlinx.dl.dataset.embedded.mnist
import org.jetbrains.kotlinx.dl.dataset.evaluate
import org.jetbrains.kotlinx.dl.dataset.predict
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import java.io.File

private const val PATH_TO_MODEL = "src/test/resources/savedmodel"

class SavedModelTest {
    @Test
    fun basicInferenceOnMnist() {
        val (train, test) = mnist()

        val modelDirectory = File(PATH_TO_MODEL)

        SavedModel.load(modelDirectory.absolutePath).use {
            val prediction = it.predict(train.getX(0))

            assertEquals(train.getY(0), prediction.toFloat())

            val predictions = it.predict(test, SavedModel::predict)

            assertEquals(10000, predictions.size)
            assertTrue(it.evaluate(test, Metrics.ACCURACY, SavedModel::predict) > 0.9)
        }
    }

    @Test
    fun basicInferenceOnMnistWithCustomTensorNames() {
        val (train, test) = mnist()

        val modelDirectory = File(PATH_TO_MODEL)

        SavedModel.load(modelDirectory.absolutePath).use {
            val prediction = it.predict(train.getX(0), "Placeholder", "ArgMax")

            assertEquals(train.getY(0), prediction.toFloat())

            val predictions = it.predict(test) { data -> predict(data, "Placeholder", "ArgMax") }

            assertEquals(10000, predictions.size)
            assertTrue(it.evaluate(test, Metrics.ACCURACY, SavedModel::predict) > 0.9)
            assertTrue(it.evaluate(test, Metrics.MAE, SavedModel::predict).isNaN())
        }
    }
}


