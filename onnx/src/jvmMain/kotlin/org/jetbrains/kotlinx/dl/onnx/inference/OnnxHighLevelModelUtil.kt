/*
 * Copyright 2023 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.onnx.inference

import org.jetbrains.kotlinx.dl.api.inference.FlatShape
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.crop
import java.awt.image.BufferedImage

/**
 * Runs prediction on an image crop, base on the provided [detectedObject]. Could be used for combining several
 * inference models together, for example detecting a face with
 * a [org.jetbrains.kotlinx.dl.onnx.inference.facealignment.FaceDetectionModel], then detecting face landmarks on the face
 * using a [org.jetbrains.kotlinx.dl.onnx.inference.facealignment.Fan2D106FaceAlignmentModel].
 *
 * @param [image] input image
 * @param [detectedObject] object, detected on this image, to run predictions on
 */
public fun <T : FlatShape<T>> OnnxHighLevelModel<BufferedImage, List<T>>.predictOnCrop(image: BufferedImage,
                                                                                       detectedObject: DetectedObject
): List<T> {
    val objectWidth = detectedObject.xMax - detectedObject.xMin
    val objectHeight = detectedObject.yMax - detectedObject.yMin
    val x1 = ((detectedObject.xMin - 0.1f * objectWidth) * image.width).toInt().coerceAtLeast(0)
    val y1 = ((detectedObject.yMin - 0.1f * objectHeight) * image.height).toInt().coerceAtLeast(0)
    val x2 = ((detectedObject.xMax + 0.1f * objectWidth) * image.width).toInt().coerceAtMost(image.width)
    val y2 = ((detectedObject.yMax + 0.1f * objectHeight) * image.height).toInt().coerceAtMost(image.height)

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