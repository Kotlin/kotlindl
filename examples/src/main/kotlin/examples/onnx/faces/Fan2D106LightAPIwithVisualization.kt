/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.faces

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.inference.facealignment.Landmark
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.api.summary.printSummary
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.ImageConverter
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.resize
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModelHub
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModels
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
    val model = ONNXModels.FaceAlignment.Fan2d106.pretrainedModel(modelHub)
    model.printSummary()

    model.use {
        val result = mutableMapOf<BufferedImage, List<Landmark>>()
        for (i in 1..8) {
            val file = getFileFromResource("datasets/faces/image$i.jpg")
            val image = ImageConverter.toBufferedImage(file)
            val landmarks = it.detectLandmarks(image)
            result[image] = landmarks
        }

        val panel = JPanel(GridLayout(2, 4))
        val resize = pipeline<BufferedImage>().resize { outputWidth = 200; outputHeight = 200 }
        for ((image, landmarks) in result) {
            panel.add(createDetectedLandmarksPanel(resize.apply(image), landmarks))
        }
        showFrame("Face Landmarks", panel)
    }
}
