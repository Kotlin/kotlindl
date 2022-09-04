/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.faces

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.facealignment.Fan2D106FaceAlignmentModel
import org.jetbrains.kotlinx.dl.dataset.image.ImageConverter
import org.jetbrains.kotlinx.dl.dataset.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
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
    val model = ONNXModels.FaceAlignment.Fan2d106.pretrainedModel(modelHub)

    model.use {
        for (i in 0..8) {
            val image = ImageConverter.toBufferedImage(getFileFromResource("datasets/faces/image$i.jpg"))
            val landmarks = it.detectLandmarks(image)
            val displayedImage = pipeline<BufferedImage>()
                .resize { outputWidth = 200; outputHeight = 200 }
                .apply(image)
            drawLandMarks(displayedImage, landmarks)
        }
    }
}
