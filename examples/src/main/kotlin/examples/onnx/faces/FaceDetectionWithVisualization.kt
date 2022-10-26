/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.faces

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.api.summary.printSummary
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.ImageConverter
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.resize
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModelHub
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModels
import org.jetbrains.kotlinx.dl.onnx.inference.executionproviders.ExecutionProvider
import org.jetbrains.kotlinx.dl.onnx.inference.inferAndCloseUsing
import org.jetbrains.kotlinx.dl.visualization.swing.createDetectedObjectsPanel
import org.jetbrains.kotlinx.dl.visualization.swing.showFrame
import java.awt.image.BufferedImage
import java.io.File

fun main() {
    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val model = ONNXModels.FaceDetection.UltraFace320.pretrainedModel(modelHub)
    model.printSummary()

    model.inferAndCloseUsing(ExecutionProvider.CPU()) {
        val file = getFileFromResource("datasets/poses/multi/1.jpg")
        val image = ImageConverter.toBufferedImage(file)
        val faces = it.detectFaces(image)

        val width = 600
        val resize = pipeline<BufferedImage>().resize {
            outputWidth = width
            outputHeight = width * image.height / image.width
        }
        showFrame("Detected Faces", createDetectedObjectsPanel(resize.apply(image), faces))
    }
}