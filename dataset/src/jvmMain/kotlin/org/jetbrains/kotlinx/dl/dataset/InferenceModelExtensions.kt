/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset

import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel

/**
 * Predicts labels for all observation in [dataset].
 *
 * NOTE: Slow method.
 *
 * @param [dataset] Dataset.
 */
public fun InferenceModel.predict(dataset: Dataset): List<Int> {
    val predictedLabels: MutableList<Int> = mutableListOf()

    for (i in 0 until dataset.xSize()) {
        val predictedLabel = predict(dataset.getX(i))
        predictedLabels.add(predictedLabel)
    }

    return predictedLabels
}

/**
 * Evaluates [dataset] via [metric].
 *
 * NOTE: Slow method.
 */
public fun InferenceModel.evaluate(
    dataset: Dataset,
    metric: Metrics
): Double {
    return if (metric == Metrics.ACCURACY) {
        var counter = 0
        for (i in 0 until dataset.xSize()) {
            val predictedLabel = predict(dataset.getX(i))
            if (predictedLabel == dataset.getY(i).toInt())
                counter++
        }

        (counter.toDouble() / dataset.xSize())
    } else {
        Double.NaN
    }
}