/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.inference.fashionmnist

import org.jetbrains.kotlinx.dl.api.inference.TensorFlowInferenceModel
import org.jetbrains.kotlinx.dl.dataset.embedded.fashionMnist
import java.io.File

private const val PATH_TO_MODEL = "savedmodels/fashionLenet"

/**
 * Inference model is used here, separately from model training code to illustrate the ability to load model graph and weights to start prediction process.
 *
 * NOTE: The example requires the saved model in the appropriate directory (run [lenetOnFashionMnistExportImportToTxt] firstly).
 */
fun main() {
    val (train, _) = fashionMnist()

    val inferenceModel = TensorFlowInferenceModel.load(File(PATH_TO_MODEL), loadOptimizerState = true)

    inferenceModel.use {
        it.reshape(28, 28, 1)

        var accuracy = 0.0
        val amountOfTestSet = 10000
        for (imageId in 0..amountOfTestSet) {
            val prediction = it.predict(train.getX(imageId))

            if (prediction == train.getY(imageId).toInt())
                accuracy += (1.0 / amountOfTestSet)
        }
        println("Accuracy: $accuracy")

        val amountOfOps = 1000
        val start = System.currentTimeMillis()
        for (i in 0..amountOfOps) {
            it.predict(train.getX(i % 50000))
        }
        println("Time, s: ${(System.currentTimeMillis() - start) / 1000f}")
        println("Throughput, op/s: ${amountOfOps / ((System.currentTimeMillis() - start) / 1000f)}")
    }
}
