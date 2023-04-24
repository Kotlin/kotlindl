/*
 * Copyright 2023 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.faces

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.inference.FlatShape
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.api.summary.printSummary
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.ImageConverter
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.crop
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.resize
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModelHub
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModels
import org.jetbrains.kotlinx.dl.onnx.inference.OnnxHighLevelModel
import org.jetbrains.kotlinx.dl.visualization.swing.createDetectedLandmarksPanel
import org.jetbrains.kotlinx.dl.visualization.swing.showFrame
import java.awt.image.BufferedImage
import java.io.File

/**
 * This example demonstrates how to combine Fan2d106 face alignment model and UltraFace320 face detection model to
 * find face landmarks on the image. The face alignment model works well only when the face has a certain size and location,
 * so for other cases it is necessary to find the face location first with a face detection model
 * and only then apply face alignment model to detect landmarks on the face crop.
 */
fun main() {
    val image = ImageConverter.toBufferedImage(getFileFromResource("datasets/poses/multi/2.jpg"))

    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))

    val detectionModel = ONNXModels.FaceDetection.UltraFace320.pretrainedModel(modelHub)
    detectionModel.printSummary()

    val faces = detectionModel.use { it.detectFaces(image) }

    val alignmentModel = ONNXModels.FaceAlignment.Fan2d106.pretrainedModel(modelHub)
    alignmentModel.printSummary()

    val facesToLandmarks = alignmentModel.use {
        faces.associateWith { face -> alignmentModel.predictOnCrop(image, face) }
    }

    val width = 600
    val resize = pipeline<BufferedImage>().resize {
        outputWidth = width
        outputHeight = width * image.height / image.width
    }
    showFrame(
        "Detected Landmarks For ${facesToLandmarks.size} Face" + (if (facesToLandmarks.size == 1) "" else "s"),
        createDetectedLandmarksPanel(resize.apply(image), facesToLandmarks.values.flatten())
    )
}

private fun <T : FlatShape<T>> OnnxHighLevelModel<BufferedImage, List<T>>.predictOnCrop(image: BufferedImage,
                                                                                        face: DetectedObject
): List<T> {
    val faceWidth = face.xMax - face.xMin
    val faceHeight = face.yMax - face.yMin
    val x1 = ((face.xMin - 0.1f * faceWidth) * image.width).toInt().coerceAtLeast(0)
    val y1 = ((face.yMin - 0.1f * faceHeight) * image.height).toInt().coerceAtLeast(0)
    val x2 = ((face.xMax + 0.1f * faceWidth) * image.width).toInt().coerceAtMost(image.width)
    val y2 = ((face.yMax + 0.1f * faceHeight) * image.height).toInt().coerceAtMost(image.height)
    val cropImage = pipeline<BufferedImage>().crop {
        left = x1
        top = y1
        right = image.width - x2
        bottom = image.height - y2

    }.apply(image)
    return predict(cropImage).map { shape ->
        shape.map { x, y ->
            (x1 + x * cropImage.width) / image.width to
                    (y1 + y * cropImage.height) / image.height
        }
    }
}