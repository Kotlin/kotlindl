/*
 * Copyright 2020-2023 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset

import org.jetbrains.kotlinx.dl.api.core.FloatData
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel

/**
 * Runs [predictionFunction] for all observations in [dataset]
 * and collects predictions to the list.
 *
 * NOTE: Slow method.
 *
 * @param [R] Prediction result type.
 * @param [T] Model type.
 * @param [dataset] Dataset.
 * @param [predictionFunction] Prediction function to make predictions with.
 */
public fun <R, T : InferenceModel<*>> T.predict(dataset: Dataset, predictionFunction: T.(FloatData) -> R): List<R> {
    return dataset.map { predictionFunction(it) }
}

/**
 * Evaluates [dataset] via [metric] with the given [predictionFunction].
 *
 * NOTE: Slow method.
 *
 * @param [T] Model type.
 * @param [dataset] Dataset.
 * @param [metric] Metric to use.
 * @param [predictionFunction] Prediction function to make predictions with.
 */
public fun <T : InferenceModel<*>> T.evaluate(dataset: Dataset,
                                              metric: Metrics,
                                              predictionFunction: T.(FloatData) -> Int
): Double {
    if (metric != Metrics.ACCURACY) return Double.NaN

    var counter = 0
    for (i in 0 until dataset.xSize()) {
        val predictedLabel = predictionFunction(dataset.getX(i))
        if (predictedLabel == dataset.getY(i).toInt())
            counter++
    }

    return (counter.toDouble() / dataset.xSize())
}