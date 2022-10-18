/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.inference.mnist

import org.jetbrains.kotlinx.dl.api.inference.TensorFlowInferenceModel
import org.jetbrains.kotlinx.dl.dataset.embedded.mnist
import java.io.File

private const val PATH_TO_MODEL = "savedmodels/lenet5"

/**
 * Inference model is used here,
 * separately from model training code to illustrate the ability
 * to load model graph and weights to start prediction process.
 *
 * After loading and evaluation, the Inference model is copied and evaluated again.
 *
 * NOTE: The example requires the saved model in the appropriate directory (run [lenetOnMnistDatasetExportImportToTxt] firstly).
 */
fun main() {
    val (train, _) = mnist()

    val inferenceModel = TensorFlowInferenceModel.load(File(PATH_TO_MODEL), loadOptimizerState = true)

    var copiedInferenceModel: TensorFlowInferenceModel

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

        copiedInferenceModel = inferenceModel.copy("CopiedLenet")
    }


    copiedInferenceModel.use {
        var accuracy = 0.0
        val amountOfTestSet = 10000
        for (imageId in 0..amountOfTestSet) {
            val prediction = it.predict(train.getX(imageId))

            if (prediction == train.getY(imageId).toInt())
                accuracy += (1.0 / amountOfTestSet)
        }
        println("Accuracy: $accuracy")
    }
}
