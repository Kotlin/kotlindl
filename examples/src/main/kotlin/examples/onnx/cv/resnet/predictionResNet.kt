/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.cv.resnet

import examples.transferlearning.modelzoo.vgg16.getFileFromResource
import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.TFModels
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.TFModelHub
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape
import org.jetbrains.kotlinx.dl.dataset.preprocessor.Preprocessing
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.load
import org.jetbrains.kotlinx.dl.dataset.preprocessor.preprocess
import org.jetbrains.kotlinx.dl.dataset.preprocessor.transformImage
import java.io.File

private const val PATH_TO_MODEL = "examples/src/main/resources/models/onnx/resnet50.onnx"

fun main() {
    val modelHub = TFModelHub(commonModelDirectory = File("cache/pretrainedModels"), modelType = TFModels.CV.ResNet_50)
    val model = modelHub.loadModel() as Functional

    val imageNetClassLabels = modelHub.loadClassLabels()

    OnnxInferenceModel.load(PATH_TO_MODEL).use {
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

            val inputData = modelHub.preprocessInput(preprocessing().first, model.inputDimensions)

            val res = it.predict(inputData)
            println("Predicted object for image$i.jpg is ${imageNetClassLabels[res]}")

            /*val top5 = predictTop5Labels(it, inputData, imageNetClassLabels)

            println(top5.toString())*/
        }
    }
}


/** Returns top-5 labels for the given [floatArray] encoded with mapping [imageNetClassLabels]. */
/*public fun predictTop5Labels(
    it: OnnxModel,
    floatArray: FloatArray,
    imageNetClassLabels: MutableMap<Int, String>
): MutableMap<Int, Pair<String, Float>> {
    val predictionVector = it.predictSoftly(floatArray).toMutableList()
    val predictionVector2 = it.predictSoftly(floatArray).toMutableList() // get copy of previous vector

    val top5: MutableMap<Int, Pair<String, Float>> = mutableMapOf()
    for (j in 1..5) {
        val max = predictionVector2.maxOrNull()
        val indexOfElem = predictionVector.indexOf(max!!)
        top5[j] = Pair(imageNetClassLabels[indexOfElem]!!, predictionVector[indexOfElem])
        predictionVector2.remove(max)
    }

    return top5
}*/

