/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.inference.onnx.inception

import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxModel
import org.jetbrains.kotlinx.dl.datasets.Dataset
import org.jetbrains.kotlinx.dl.datasets.handlers.*

private const val PATH_TO_MODEL = "api/src/main/resources/models/onnx/inception-v1-9.onnx"

fun main() {
    val (train, test) = Dataset.createTrainAndTestDatasets(
        FASHION_TRAIN_IMAGES_ARCHIVE,
        FASHION_TRAIN_LABELS_ARCHIVE,
        FASHION_TEST_IMAGES_ARCHIVE,
        FASHION_TEST_LABELS_ARCHIVE,
        NUMBER_OF_CLASSES,
        ::extractFashionImages,
        ::extractFashionLabels
    )

    OnnxModel.load(PATH_TO_MODEL).use {
        println(it)

        it.reshape(::reshapeInput)

        val prediction = it.predict(train.getX(0))

        println("Predicted Label is: $prediction")
        println("Correct Label is: " + train.getLabel(0))

        val predictions = it.predictAll(test)
        println(predictions.toString())

        println("Accuracy is : ${it.evaluate(test, Metrics.ACCURACY)}")
    }
}

// need good dataset for inception and another NN 3*224*224
fun reshapeInput(inputData: FloatArray): Array<Array<Array<FloatArray>>> {
    val reshaped = Array(1) {
        Array(3)
        { Array(224) { FloatArray(224) } }
    }

    for (i in inputData.indices) reshaped[0][0][(i / 28) * 8][(i % 28) * 8] = inputData[i]
    for (i in inputData.indices) reshaped[0][1][(i / 28) * 8][(i % 28) * 8] = inputData[i]
    for (i in inputData.indices) reshaped[0][2][(i / 28) * 8][(i % 28) * 8] = inputData[i]
    return reshaped
}
