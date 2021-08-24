/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras.loaders

import com.beust.klaxon.JsonArray
import com.beust.klaxon.JsonObject
import com.beust.klaxon.Parser
import org.jetbrains.kotlinx.dl.api.core.shape.reshape3DTo1D
import org.jetbrains.kotlinx.dl.api.inference.TensorFlowInferenceModel

/**
 * Common preprocessing functions for the Neural Networks trained on ImageNet and whose weights are available with the keras.application.
 *
 * It takes [floatArray] as input with shape [tensorShape] and calls specific preprocessing according chosen [inputType].
 */
public fun preprocessInput(floatArray: FloatArray, tensorShape: LongArray? = null, inputType: InputType): FloatArray {
    return when (inputType) {
        InputType.TF -> floatArray.map { it / 127.5f - 1 }.toFloatArray()
        InputType.CAFFE -> caffeStylePreprocessing(floatArray, tensorShape!!)
        InputType.TORCH -> torchStylePreprocessing(floatArray, tensorShape!!)
    }
}

/** Torch-style preprocessing. */
public fun torchStylePreprocessing(input: FloatArray, tensorShape: LongArray): FloatArray {
    val height = tensorShape[0].toInt()
    val width = tensorShape[1].toInt()
    val channels = tensorShape[2].toInt()

    val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
    val std = floatArrayOf(0.229f, 0.224f, 0.225f)

    val scaledInput = input.map { it / 255f }.toFloatArray()
    val reshapedInput = reshapeInput(scaledInput, tensorShape)

    for (i in 0 until height) {
        for (j in 0 until width) {
            for (k in 0 until channels) {
                reshapedInput[i][j][k] = (reshapedInput[i][j][k] - mean[k]) / std[k]
            }
        }
    }

    return reshape3DTo1D(reshapedInput, height * width * channels)
}

/** Caffe-style preprocessing. */
public fun caffeStylePreprocessing(input: FloatArray, tensorShape: LongArray): FloatArray {
    val height = tensorShape[0].toInt()
    val width = tensorShape[1].toInt()
    val channels = tensorShape[2].toInt()

    val mean = floatArrayOf(103.939f, 116.779f, 123.68f)

    val reshapedInput = reshapeInput(input, tensorShape)

    for (i in 0 until height) {
        for (j in 0 until width) {
            for (k in 0 until channels) {
                reshapedInput[i][j][k] = reshapedInput[i][j][k] - mean[k]
            }
        }
    }

    return reshape3DTo1D(reshapedInput, height * width * channels)
}

/** Reshapes [inputData] according [tensorShape]. */
public fun reshapeInput(inputData: FloatArray, tensorShape: LongArray): Array<Array<FloatArray>> {
    val height = tensorShape[0].toInt()
    val width = tensorShape[1].toInt()
    val channels = tensorShape[2].toInt()
    val reshaped = Array(
        height
    ) { Array(width) { FloatArray(channels) } }

    var pos = 0
    for (i in 0 until height) {
        for (j in 0 until width) {
            for (k in 0 until channels) {
                reshaped[i][j][k] = inputData[pos]
                pos++
            }
        }
    }

    return reshaped
}

/** Returns top-5 labels for the given [floatArray] encoded with mapping [imageNetClassLabels]. */
public fun predictTop5ImageNetLabels(
    it: TensorFlowInferenceModel,
    floatArray: FloatArray,
    imageNetClassLabels: MutableMap<Int, String>
): MutableMap<Int, Pair<String, Float>> {
    val predictionVector = it.predictSoftly(floatArray).toMutableList()
    val predictionVector2 = it.predictSoftly(floatArray).toMutableList() // get copy of previous vector

    val top5: MutableMap<Int, Pair<String, Float>> = mutableMapOf()
    for (j in 1..5) {
        val max = predictionVector2.maxOrNull()
        val indexOfElem = predictionVector.indexOf(max!!)
        top5[j] = Pair(imageNetClassLabels[indexOfElem]!!, predictionVector[indexOfElem])
        predictionVector2.remove(max)
    }

    return top5
}

/** Forms mapping of class label to class name for the ImageNet dataset. */
public fun prepareImageNetHumanReadableClassLabels(): MutableMap<Int, String> {
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
