/*
 * Copyright 2020-2023 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.inference.imagerecognition

import org.jetbrains.kotlinx.dl.api.core.FloatData
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.impl.util.argmax
import java.util.*

/**
 * Predicts the class of [inputData].
 *
 * @param [inputData] The single example with unknown label.
 * @return Predicted class index.
 */
public fun <R> InferenceModel<R>.predictLabel(inputData: FloatData): Int {
    return predictProbabilities(inputData).argmax()
}

/**
 * Predicts vector of probabilities instead of specific class in [predictLabel] method.
 *
 * @param [inputData] The single example with unknown vector of probabilities.
 * @return Vector that represents the probability distributions of possible outcomes.
 */
public fun <R> InferenceModel<R>.predictProbabilities(inputData: FloatData): FloatArray {
    return predict(inputData) { result ->
        resultConverter.getFloatArray(result, 0)
    }
}

/** Returns top-N labels for the given [floatData] encoded with mapping [labels]. */
public fun InferenceModel<*>.predictTopNLabels(
    floatData: FloatData,
    labels: Map<Int, String>,
    n: Int = 5
): List<Pair<String, Float>> {
    val prediction = predictProbabilities(floatData)
    val topNIndexes = prediction.indexOfMaxN(n)
    return topNIndexes.map { index -> labels[index]!! to prediction[index] }
}

/** Returns top-5 labels for the given [data] encoded with mapping [classLabels]. */
public fun InferenceModel<*>.predictTop5Labels(
    data: FloatData,
    classLabels: Map<Int, String>,
): List<Pair<String, Float>> {
    return predictTopNLabels(data, classLabels, n = 5)
}

internal fun FloatArray.indexOfMaxN(n: Int): List<Int> {
    val predictionQueue = PriorityQueue<Int>(Comparator.comparing { index -> -this[index] })
    predictionQueue.addAll(indices)
    return predictionQueue.take(n)
}