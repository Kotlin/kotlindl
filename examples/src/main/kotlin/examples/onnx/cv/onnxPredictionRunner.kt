/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.cv

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.core.util.loadImageNetClassLabels
import org.jetbrains.kotlinx.dl.api.core.util.predictTopNLabels
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import java.io.File

fun runONNXImageRecognitionPrediction(
    modelType: ONNXModels.CV<out OnnxInferenceModel>,
    resizeTo: Pair<Int, Int> = Pair(224, 224)
) {
    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val model = modelHub.loadModel(modelType)

    val imageNetClassLabels = loadImageNetClassLabels()

    model.use {
        println(it)

        for (i in 1..8) {
            val preprocessing: Preprocessing = preprocessing(resizeTo, i)

            val inputData = modelType.preprocessInput(preprocessing().first, model.inputDimensions)

            val res = it.predict(inputData)
            println("Predicted object for image$i.jpg is ${imageNetClassLabels[res]}")

            val top5 = predictTopNLabels(it, inputData, imageNetClassLabels)

            println(top5.toString())
        }
    }
}

// TODO: copy-paste from predictionRunner (refactor it)
private fun preprocessing(
    resizeTo: Pair<Int, Int>,
    i: Int
): Preprocessing {
    val preprocessing: Preprocessing = if (resizeTo.first == 224 && resizeTo.second == 224) {
        preprocess {
            load {
                pathToData = getFileFromResource("datasets/vgg/image$i.jpg")
                imageShape = ImageShape(224, 224, 3)
            }
            transformImage { convert { colorMode = ColorMode.BGR } }
        }
    } else {
        preprocess {
            load {
                pathToData = getFileFromResource("datasets/vgg/image$i.jpg")
                imageShape = ImageShape(224, 224, 3)
            }
            transformImage {
                resize {
                    outputWidth = resizeTo.first
                    outputHeight = resizeTo.second
                }
                convert { colorMode = ColorMode.BGR }
            }
        }
    }
    return preprocessing
}
