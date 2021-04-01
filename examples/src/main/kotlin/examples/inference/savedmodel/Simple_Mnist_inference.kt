/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.inference.savedmodel

import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.inference.savedmodel.Input
import org.jetbrains.kotlinx.dl.api.inference.savedmodel.Output
import org.jetbrains.kotlinx.dl.api.inference.savedmodel.SavedModel
import org.jetbrains.kotlinx.dl.dataset.mnist

private const val PATH_TO_MODEL = "examples/src/main/resources/models/savedmodel"

fun main() {
    val (train, test) = mnist()

    SavedModel.load(PATH_TO_MODEL).use {
        println(it)

        it.reshape(::reshapeInput) // TODO: maybe pass here a Shape like in Sequential model, shape could a be set via setter or passed in predict method
        it.input(Input.PLACEHOLDER)
        it.output(Output.ARGMAX)

        val prediction = it.predict(train.getX(0), "Placeholder", "ArgMax")

        println("Predicted Label is: $prediction")
        println("Correct Label is: " + train.getY(0))

        val predictions = it.predictAll(test)
        println(predictions.toString())

        println("Accuracy is : ${it.evaluate(test, Metrics.ACCURACY)}")
    }
}

fun reshapeInput(inputData: FloatArray): Array<Array<FloatArray>> {
    val reshaped = Array(
        1
    ) { Array(28) { FloatArray(28) } }
    for (i in inputData.indices) reshaped[0][i / 28][i % 28] = inputData[i]
    return reshaped
}
