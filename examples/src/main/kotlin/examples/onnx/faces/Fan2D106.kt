/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.faces

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.inference.facealignment.Landmark
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.facealignment.Fan2D106FaceAlignmentModel
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.preprocessing.call
import org.jetbrains.kotlinx.dl.dataset.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.dataset.preprocessing.rescale
import org.jetbrains.kotlinx.dl.dataset.preprocessor.fileLoader
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.toFloatArray
import org.jetbrains.kotlinx.dl.dataset.preprocessor.toImageShape
import org.jetbrains.kotlinx.dl.visualization.swing.drawLandMarks
import java.awt.image.BufferedImage
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

        val fileDataLoader = pipeline<BufferedImage>()
            .resize {
                outputHeight = 192
                outputWidth = 192
            }
            .convert { colorMode = ColorMode.BGR }
            .toFloatArray { }
            .call(modelType.preprocessor)
            .fileLoader()

        for (i in 0..8) {
            val imageFile = getFileFromResource("datasets/faces/image$i.jpg")

            val inputData = fileDataLoader.load(imageFile).first

            val yhat = it.predictRaw(inputData)
            println(yhat.values.toTypedArray().contentDeepToString())

            val landMarks = mutableListOf<Landmark>()
            val floats = (yhat["fc1"] as Array<*>)[0] as FloatArray
            for (j in floats.indices step 2) {
                landMarks.add(Landmark((1 + floats[j]) / 2, (1 + floats[j + 1]) / 2))
            }

            visualiseLandMarks(imageFile, landMarks)
        }
    }
}

fun visualiseLandMarks(
    imageFile: File,
    landmarks: List<Landmark>
) {
    val preprocessing = pipeline<BufferedImage>()
        .resize {
            outputWidth = 192
            outputHeight = 192
        }
        .convert { colorMode = ColorMode.BGR }
        .toFloatArray { }
        .rescale {
            scalingCoefficient = 255f
        }

    val (rawImage, shape) = preprocessing.fileLoader().load(imageFile)
    drawLandMarks(rawImage, shape.toImageShape(), landmarks)
}
