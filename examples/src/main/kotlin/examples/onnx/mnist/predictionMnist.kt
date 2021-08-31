/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.mnist

import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.dataset.mnist
import java.io.File

fun main() {
    val (train, test) = mnist()
    val modelHub =
        ONNXModelHub(
            commonModelDirectory = File("cache/pretrainedModels"),
            modelType = ONNXModels.CV.Lenet_mnist
        )
    val model = modelHub.loadModel() as OnnxInferenceModel
    model.use {
        println(it)

        val prediction = it.predict(train.getX(0))

        println("Predicted Label is: $prediction")
        println("Correct Label is: " + train.getY(0))
        println("Accuracy is: ${it.evaluate(test, Metrics.ACCURACY)}")
    }
}
