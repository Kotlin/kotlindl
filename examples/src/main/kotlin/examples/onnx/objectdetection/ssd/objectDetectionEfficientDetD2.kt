/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.objectdetection.ssd

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.objectdetection.SSDObjectDetectionModel
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.*
import org.jetbrains.kotlinx.dl.visualization.swing.drawDetectedObjects
import java.io.File

/**
 * This examples demonstrates the light-weight inference API with [SSDObjectDetectionModel] on SSD model:
 * - Model is obtained from [ONNXModelHub].
 * - Model predicts rectangles for the detected objects on a few images located in resources.
 * - The detected rectangles related to the objects are drawn on the images used for prediction.
 */
fun main() {
    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val modelType = ONNXModels.ObjectDetection.EfficientDetD2 // TODO: input/output https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/efficientdet.ipynb
    val model = modelHub.loadModel(modelType)

    model.use {
        it.inputShape = longArrayOf(1, 1920, 1280, 3)
        println(it)

        for (i in 0..9) {
            val preprocessing: Preprocessing = preprocess {
                load {
                    pathToData = getFileFromResource("datasets/vgg/image$i.jpg")
                    imageShape = ImageShape(224, 224, 3)
                }
                transformImage {
                    resize {
                        outputHeight = 1920
                        outputWidth = 1280
                    }
                    convert { colorMode = ColorMode.BGR }
                }
            }

            val inputData = modelType.preprocessInput(preprocessing)

            val yhat = it.predictRaw(inputData)
            println(yhat.toTypedArray().contentDeepToString())
        }
    }
}


