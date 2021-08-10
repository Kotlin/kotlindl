/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx

import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.dataset.mnist

private const val PATH_TO_MODEL = "examples/src/main/resources/models/onnx/mnist-8.onnx"

fun main() {
    val (train, test) = mnist()

    OnnxInferenceModel.load(PATH_TO_MODEL).use {
        println(it)

        it.reshape(1, 28, 28)

        val prediction = it.predict(train.getX(0))

        println("Predicted Label is: $prediction")
        println("Correct Label is: " + train.getY(0))
    }
}

fun reshapeInput(inputData: FloatArray): Array<Array<Array<FloatArray>>> {
    val reshaped = Array(1) {
        Array(1)
        { Array(28) { FloatArray(28) } }
    }

    for (i in inputData.indices) reshaped[0][0][i / 28][i % 28] = inputData[i]
    return reshaped
}
