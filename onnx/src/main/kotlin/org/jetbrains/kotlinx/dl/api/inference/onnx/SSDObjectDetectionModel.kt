/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.onnx

import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.dataset.handler.cocoCategories
import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape
import org.jetbrains.kotlinx.dl.dataset.preprocessor.Preprocessing
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.load
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import org.jetbrains.kotlinx.dl.dataset.preprocessor.preprocess
import org.jetbrains.kotlinx.dl.dataset.preprocessor.transformImage
import java.io.File

class SSDObjectDetectionModel : OnnxInferenceModel() {
    public fun detectObjects(inputData: FloatArray, topK: Int = 5): List<DetectedObject> {
        val rawPrediction = this.predictRaw(inputData)

        val foundObjects = mutableListOf<DetectedObject>()
        val boxes = rawPrediction[0][0] as Array<FloatArray>
        val classIndices = rawPrediction[1][0] as LongArray
        val probabilities = rawPrediction[2][0] as FloatArray
        val numberOfFoundObjects = boxes.size

        for (i in 0 until numberOfFoundObjects) {
            val detectedObject = DetectedObject(
                classLabel = cocoCategories[classIndices[i].toInt()]!!,
                probability = probabilities[i],
                // left, bot, right, top
                xMin = boxes[i][0],
                yMax = boxes[i][1],
                xMax = boxes[i][2],
                yMin = boxes[i][3]
            )
            foundObjects.add(detectedObject)
        }

        if (topK > 0) {
            foundObjects.sortByDescending { it.probability }
            return foundObjects.take(topK)
        }

        return foundObjects
    }

    public fun detectObjects(imageFile: File, topK: Int = 5): List<DetectedObject> {
        val preprocessing: Preprocessing = preprocess {
            transformImage {
                load {
                    pathToData = imageFile
                    imageShape = ImageShape(224, 224, 3)
                    colorMode = ColorOrder.BGR
                }
                resize {
                    outputHeight = 1200
                    outputWidth = 1200
                }
            }
        }

        val (data, shape) = preprocessing()

        val preprocessedData = ONNXModels.ObjectDetection.SSD.preprocessInput(
            data,
            longArrayOf(1, shape.width!!, shape.height!!, shape.channels)
        )

        return this.detectObjects(preprocessedData, topK)
    }
}
