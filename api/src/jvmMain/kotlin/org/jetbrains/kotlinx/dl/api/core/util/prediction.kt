/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.util

import com.beust.klaxon.JsonArray
import com.beust.klaxon.JsonObject
import com.beust.klaxon.Parser
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel

/** Returns top-N labels for the given [floatArray] encoded with mapping [labels]. */
public fun predictTopNLabels(
    it: InferenceModel,
    floatArray: FloatArray,
    labels: Map<Int, String>,
    topN: Int = 5
): Map<Int, Pair<String, Float>> {
    val predictionVector = it.predictSoftly(floatArray).toMutableList()
    val predictionVector2 =
        it.predictSoftly(floatArray).toMutableList() //NOTE: don't remove this row, it gets a copy of previous vector

    check(predictionVector.size >= topN) { "TopN should be less or equal than ${predictionVector.size}." }

    val top5: MutableMap<Int, Pair<String, Float>> = mutableMapOf()
    for (j in 1..topN) {
        val max = predictionVector2.maxOrNull()
        val indexOfElem = predictionVector.indexOf(max!!)
        top5[j] = Pair(labels[indexOfElem]!!, predictionVector[indexOfElem])
        predictionVector2.remove(max)
    }

    return top5
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
