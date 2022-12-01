/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.faces

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.inference.facealignment.Landmark
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.api.summary.printSummary
import org.jetbrains.kotlinx.dl.impl.preprocessing.call
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.*
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModelHub
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModels
import org.jetbrains.kotlinx.dl.onnx.inference.OrtSessionResultConversions.getFloatArray
import org.jetbrains.kotlinx.dl.onnx.inference.facealignment.Fan2D106FaceAlignmentModel
import org.jetbrains.kotlinx.dl.visualization.swing.createDetectedLandmarksPanel
import org.jetbrains.kotlinx.dl.visualization.swing.showFrame
import java.awt.GridLayout
import java.awt.image.BufferedImage
import java.io.File
import javax.swing.JPanel

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
    model.printSummary()

    model.use {
        println(it)

        val preprocessor = pipeline<BufferedImage>()
            .resize {
                outputHeight = 192
                outputWidth = 192
            }
            .convert { colorMode = ColorMode.BGR }
            .toFloatArray { }
            .call(modelType.preprocessor)

        val result = mutableMapOf<BufferedImage, List<Landmark>>()
        for (i in 1..8) {
            val inputFile = getFileFromResource("datasets/faces/image$i.jpg")
            val inputImage = ImageConverter.toBufferedImage(inputFile)
            val inputData = preprocessor.apply(inputImage)

            val floats = it.predictRaw(inputData) { output -> output.getFloatArray("fc1") }
            println(floats.contentToString())

            val landMarks = mutableListOf<Landmark>()
            for (j in floats.indices step 2) {
                landMarks.add(Landmark((1 + floats[j]) / 2, (1 + floats[j + 1]) / 2))
            }
            result[inputImage] = landMarks
        }

        val panel = JPanel(GridLayout(2, 4))
        val resize = pipeline<BufferedImage>().resize { outputWidth = 200; outputHeight = 200 }
        for ((image, landmarks) in result) {
            panel.add(createDetectedLandmarksPanel(resize.apply(image), landmarks))
        }
        showFrame("Face Landmarks", panel)
    }
}
