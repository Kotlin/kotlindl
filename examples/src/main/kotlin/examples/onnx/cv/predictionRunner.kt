/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.cv

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.util.loadImageNetClassLabels
import org.jetbrains.kotlinx.dl.api.core.util.predictTopNLabels
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeights
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.TFModelHub
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.TFModels
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.predictTop5ImageNetLabels
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape
import org.jetbrains.kotlinx.dl.dataset.preprocessor.Preprocessing
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.load
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import org.jetbrains.kotlinx.dl.dataset.preprocessor.preprocess
import org.jetbrains.kotlinx.dl.dataset.preprocessor.transformImage
import java.io.File

fun runImageRecognitionPrediction(
    modelType: ONNXModels.CV<OnnxInferenceModel>
) {
    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val model = modelHub.loadModel(modelType)

    val imageNetClassLabels =
        loadImageNetClassLabels()

    model.use {
        println(it)

        for (i in 1..8) {
            val preprocessing: Preprocessing = preprocess {
                transformImage {
                    load {
                        pathToData = getFileFromResource("datasets/vgg/image$i.jpg")
                        imageShape = ImageShape(224, 224, 3)
                        colorMode = ColorOrder.BGR
                    }
                }
            }

            val inputData = modelType.preprocessInput(preprocessing)

            val res = it.predict(inputData)
            println("Predicted object for image$i.jpg is ${imageNetClassLabels[res]}")

            val top5 = predictTopNLabels(it, inputData, imageNetClassLabels)

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