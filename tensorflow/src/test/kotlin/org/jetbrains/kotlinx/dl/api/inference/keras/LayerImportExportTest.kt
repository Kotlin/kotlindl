/*
 * Copyright 2021-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras

import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.junit.jupiter.api.Assertions

object LayerImportExportTest {
    internal fun run(originalModel: Sequential) {
        val kerasModel = originalModel.serializeModel(false)
        val restoredModel = deserializeSequentialModel(kerasModel)
        assertHasSameLayers(originalModel, restoredModel)
    }

    internal fun run(originalModel: Functional) {
        val kerasModel = originalModel.serializeModel(false)
        val restoredModel = deserializeFunctionalModel(kerasModel)
        assertHasSameLayers(originalModel, restoredModel)
    }

    private fun assertHasSameLayers(expectedModel: GraphTrainableModel, actualModel: GraphTrainableModel) {
        listOf(expectedModel, actualModel).forEach {
            it.compile(Adam(), Losses.MSE, Metrics.ACCURACY)
        }
        Assertions.assertEquals(
            expectedModel.layers.joinToString("\n"),
            actualModel.layers.joinToString("\n")
        )
    }
}