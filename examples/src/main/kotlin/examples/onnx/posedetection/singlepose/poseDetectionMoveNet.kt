/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.posedetection.singlepose

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import org.jetbrains.kotlinx.dl.visualization.swing.drawRawPoseLandMarks
import java.io.File

/**
 * This examples demonstrates the inference concept on MoveNetSinglePoseLighting model:
 * - Model is obtained from [ONNXModelHub].
 * - Model predicts on a few images located in resources.
 * - Special preprocessing is applied to images before prediction.
 */
fun poseDetectionMoveNet() {
    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val modelType = ONNXModels.PoseEstimation.MoveNetSinglePoseLighting
    val model = modelHub.loadModel(modelType)

    model.use {
        println(it)

        val imageFile = getFileFromResource("datasets/poses/single/3.jpg")
        val preprocessing: Preprocessing = preprocess {
            load {
                pathToData = imageFile
                imageShape = ImageShape(null, null, 3)
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

        val rawPoseLandMarks = (yhat["output_0"] as Array<Array<Array<FloatArray>>>)[0][0]

        // Dictionary that maps from joint names to keypoint indices.
        val keypoints = mapOf(
            0 to "nose",
            1 to "left_eye",
            2 to "right_eye",
            3 to "left_ear",
            4 to "right_ear",
            5 to "left_shoulder",
            6 to "right_shoulder",
            7 to "left_elbow",
            8 to "right_elbow",
            9 to "left_wrist",
            10 to "right_wrist",
            11 to "left_hip",
            12 to "right_hip",
            13 to "left_knee",
            14 to "right_knee",
            15 to "left_ankle",
            16 to "right_ankle"
        )

        rawPoseLandMarks.forEachIndexed { index, data ->
            println(keypoints[index] + " x = " + data[1] + " y =  " + data[0] + " score = " + data[2])
        }

        visualisePoseLandmarks(imageFile, rawPoseLandMarks)
    }
}

private fun visualisePoseLandmarks(
    imageFile: File,
    poseLandmarks: Array<FloatArray>
) {
    val preprocessing: Preprocessing = preprocess {
        load {
            pathToData = imageFile
            imageShape = ImageShape(null, null, 3)
        }
        transformImage {
            resize {
                outputHeight = 256
                outputWidth = 256
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
    drawRawPoseLandMarks(rawImage, ImageShape(256, 256, 3), poseLandmarks)
}

/** */
fun main(): Unit = poseDetectionMoveNet()

