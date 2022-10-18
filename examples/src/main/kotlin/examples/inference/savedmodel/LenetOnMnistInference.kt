/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.inference.savedmodel

import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.inference.savedmodel.Input
import org.jetbrains.kotlinx.dl.api.inference.savedmodel.Output
import org.jetbrains.kotlinx.dl.api.inference.savedmodel.SavedModel
import org.jetbrains.kotlinx.dl.dataset.embedded.mnist
import org.jetbrains.kotlinx.dl.dataset.evaluate
import org.jetbrains.kotlinx.dl.dataset.predict

private const val PATH_TO_MODEL = "examples/src/main/resources/models/savedmodel"

/**
 * This examples demonstrates running [SavedModel] for prediction on [mnist] dataset.
 *
 * It uses enum-based tensor names to get access to input/output tensors in TensorFlow static graph.
 */
fun lenetOnMnistInference() {
    val (train, test) = mnist()

    SavedModel.load(PATH_TO_MODEL).use {
        println(it.kGraph.toString())

        it.reshape(28, 28, 1)
        it.input(Input.PLACEHOLDER)
        it.output(Output.ARGMAX)

        val prediction = it.predict(train.getX(0), "Placeholder", "ArgMax")

        println("Predicted Label is: $prediction")
        println("Correct Label is: " + train.getY(0))

        val predictions = it.predict(test)
        println(predictions.toString())

        println("Accuracy is : ${it.evaluate(test, Metrics.ACCURACY)}")
    }
}

/** */
fun main(): Unit = lenetOnMnistInference()

