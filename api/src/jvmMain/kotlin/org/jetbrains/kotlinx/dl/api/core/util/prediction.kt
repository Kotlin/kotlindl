/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.util

import com.beust.klaxon.JsonArray
import com.beust.klaxon.JsonObject
import com.beust.klaxon.Parser
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import java.util.*

/** Returns top-N labels for the given [floatArray] encoded with mapping [labels]. */
public fun InferenceModel.predictTopNLabels(
    floatArray: FloatArray,
    labels: Map<Int, String>,
    n: Int = 5
): List<Pair<String, Float>> {
    val prediction = predictSoftly(floatArray)
    val topNIndexes = prediction.indexOfMaxN(n)
    return topNIndexes.map { index -> labels[index]!! to prediction[index] }
}

/** Returns top-5 labels for the given [data] encoded with mapping [classLabels]. */
public fun InferenceModel.predictTop5Labels(
    data: FloatArray,
    classLabels: Map<Int, String>,
): List<Pair<String, Float>> {
    return predictTopNLabels(data, classLabels, n = 5)
}

internal fun FloatArray.indexOfMaxN(n: Int): List<Int> {
    val predictionsQueue = PriorityQueue<Int>(Comparator.comparing { index -> -this[index] })
    predictionsQueue.addAll(indices)
    return predictionsQueue.take(n)
}

/** Forms mapping of class label to class name for the ImageNet dataset. */
public fun loadImageNetClassLabels(): Map<Int, String> {
    val pathToIndices = "/datasets/vgg/imagenet_class_index.json"

    fun parse(name: String): Any? {
        val cls = Parser::class.java
        return cls.getResourceAsStream(name)?.let { inputStream ->
            return Parser.default().parse(inputStream, Charsets.UTF_8)
        }
    }

    val classIndices = parse(pathToIndices) as JsonObject

    val imageNetClassIndices = mutableMapOf<Int, String>()

    for (key in classIndices.keys) {
        imageNetClassIndices[key.toInt()] = (classIndices[key] as JsonArray<*>)[1].toString()
    }
    return imageNetClassIndices
}
