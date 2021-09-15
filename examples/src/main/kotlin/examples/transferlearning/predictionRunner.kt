/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning

import org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeights
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.TFModelHub
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.TFModels
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.predictTop5ImageNetLabels
import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape
import org.jetbrains.kotlinx.dl.dataset.preprocessor.Preprocessing
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.load
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import org.jetbrains.kotlinx.dl.dataset.preprocessor.preprocess
import org.jetbrains.kotlinx.dl.dataset.preprocessor.transformImage
import java.io.File
import java.net.URISyntaxException
import java.net.URL

/** Converts resource string path to the file. */
@Throws(URISyntaxException::class)
fun getFileFromResource(fileName: String): File {
    val classLoader: ClassLoader = object {}.javaClass.classLoader
    val resource: URL? = classLoader.getResource(fileName)
    return if (resource == null) {
        throw IllegalArgumentException("file not found! $fileName")
    } else {
        File(resource.toURI())
    }
}

fun runImageRecognitionPrediction(
    modelType: TFModels.CV<out GraphTrainableModel>,
    resizeTo: Pair<Int, Int> = Pair(224, 224)
) {
    val modelHub = TFModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val model = modelHub.loadModel(modelType)

    val imageNetClassLabels = modelHub.loadClassLabels()

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.MAE,
            metric = Metrics.ACCURACY
        )

        it.summary()

        val hdfFile = modelHub.loadWeights(modelType)

        it.loadWeights(hdfFile)

        for (i in 1..8) {
            val preprocessing: Preprocessing = preprocessing(resizeTo, i)

            val inputData = modelType.preprocessInput(preprocessing().first, model.inputDimensions)

            val res = it.predict(inputData, "Activation_predictions")
            println("Predicted object for image$i.jpg is ${imageNetClassLabels[res]}")

            val top5 = predictTop5ImageNetLabels(it, inputData, imageNetClassLabels)

            println(top5.toString())
        }
    }
}

private fun preprocessing(
    resizeTo: Pair<Int, Int>,
    i: Int
): Preprocessing {
    val preprocessing: Preprocessing = if (resizeTo.first == 224 && resizeTo.second == 224) {
        preprocess {
            transformImage {
                load {
                    pathToData = getFileFromResource("datasets/vgg/image$i.jpg")
                    imageShape = ImageShape(224, 224, 3)
                    colorMode = ColorOrder.BGR
                }
            }
        }
    } else {
        preprocess {
            transformImage {
                load {
                    pathToData = getFileFromResource("datasets/vgg/image$i.jpg")
                    imageShape = ImageShape(224, 224, 3)
                    colorMode = ColorOrder.RGB
                }
                resize {
                    outputWidth = 299
                    outputHeight = 299
                }
            }
        }
    }
    return preprocessing
}
