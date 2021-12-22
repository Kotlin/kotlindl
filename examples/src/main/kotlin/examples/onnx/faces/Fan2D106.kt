/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.faces

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.facealignment.Fan2D106FaceAlignmentModel
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import org.jetbrains.kotlinx.dl.visualization.swing.drawRawLandMarks
import java.io.File

/**
 * This examples demonstrates the light-weight inference API with [Fan2D106FaceAlignmentModel] on Fan2d106 model:
 * - Model is obtained from [ONNXModelHub].
 * - Model predicts landmarks on a few images located in resources.
 * - The detected landmarks are drawn on the images used for prediction.
 */
fun main() {
    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val modelType = ONNXModels.FaceAlignment.Fan2d106
    val model = modelHub.loadModel(modelType)

    model.use {
        println(it)

        for (i in 0..8) {
            val imageFile = getFileFromResource("datasets/faces/image$i.jpg")
            val preprocessing: Preprocessing = preprocess {
                load {
                    pathToData = imageFile
                    imageShape = ImageShape(224, 224, 3)
                }
                transformImage {
                    resize {
                        outputHeight = 192
                        outputWidth = 192
                    }
                    convert { colorMode = ColorMode.BGR }
                }
            }

            val inputData = modelType.preprocessInput(preprocessing)

            val yhat = it.predictRaw(inputData)
            println(yhat.values.toTypedArray().contentDeepToString())

            visualiseLandMarks(imageFile, yhat)
        }
    }
}

fun visualiseLandMarks(
    imageFile: File,
    landmarks: Map<String, Any>
) {
    val preprocessing: Preprocessing = preprocess {
        load {
            pathToData = imageFile
            imageShape = ImageShape(224, 224, 3)
        }
        transformImage {
            resize {
                outputWidth = 192
                outputHeight = 192
            }
            convert { colorMode = ColorMode.BGR }
        }
        transformTensor {
            rescale {
                scalingCoefficient = 255f
            }
        }
    }

    val rawImage = preprocessing().first

    drawRawLandMarks(rawImage, ImageShape(192, 192, 3), landmarks)
}
