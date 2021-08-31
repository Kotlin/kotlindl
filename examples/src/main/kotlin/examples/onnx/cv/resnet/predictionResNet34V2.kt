/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.cv.resnet

import examples.transferlearning.modelzoo.vgg16.getFileFromResource
import org.jetbrains.kotlinx.dl.api.core.util.loadImageNetClassLabels
import org.jetbrains.kotlinx.dl.api.core.util.predictTopNLabels
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape
import org.jetbrains.kotlinx.dl.dataset.preprocessor.Preprocessing
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.load
import org.jetbrains.kotlinx.dl.dataset.preprocessor.preprocess
import org.jetbrains.kotlinx.dl.dataset.preprocessor.transformImage
import java.io.File

/**
 * This examples demonstrates the inference concept on ResNet'34v2 model:
 * - Model configuration, model weights and labels are obtained from [ONNXModelHub].
 * - Model predicts on a few images located in resources.
 * - Special preprocessing (used in ResNet'34v2 during training on ImageNet dataset) is applied to images before prediction.
 */
fun resnet34v2prediction() {
    val modelHub = ONNXModelHub(
        commonModelDirectory = File("cache/pretrainedModels"),
        modelType = ONNXModels.CV.ResNet_34_v2
    )
    val model = modelHub.loadModel() as OnnxInferenceModel

    println(model.inputDimensions.contentToString())

    val imageNetClassLabels =
        loadImageNetClassLabels() // TODO: move to overridden method of ModelType (loading of labels for each model)

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

            // TODO: currently, the whole model is loaded but not used for prediction, the preprocessing is used only
            val inputData = modelHub.preprocessInput(preprocessing)

            val res = it.predict(inputData)
            println("Predicted object for image$i.jpg is ${imageNetClassLabels[res]}")

            val top5 = predictTopNLabels(it, inputData, imageNetClassLabels)

            println(top5.toString())
        }
    }
}

/** */
fun main(): Unit = resnet34v2prediction()

