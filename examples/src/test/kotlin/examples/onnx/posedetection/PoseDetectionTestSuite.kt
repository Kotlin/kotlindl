/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.posedetection

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.dataset.preprocessing.fileLoader
import org.jetbrains.kotlinx.dl.impl.preprocessing.call
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.ColorMode
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.convert
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.resize
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.toFloatArray
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModelHub
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModels
import org.jetbrains.kotlinx.dl.onnx.inference.OrtSessionResultConversions.get2DFloatArray
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import java.awt.image.BufferedImage
import java.io.File

class PoseDetectionTestSuite {
    @Test
    fun easyPoseDetectionMoveNetSinglePoseLightingTest() {
        val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
        val model = ONNXModels.PoseDetection.MoveNetSinglePoseLighting.pretrainedModel(modelHub)

        model.use { poseDetectionModel ->
            val imageFile = getFileFromResource("datasets/poses/single/1.jpg")
            val detectedPose = poseDetectionModel.detectPose(imageFile = imageFile)
            assertEquals(17, detectedPose.landmarks.size)
            assertEquals(18, detectedPose.edges.size)
        }
    }

    @Test
    fun easyPoseDetectionMoveNetSinglePoseThunderTest() {
        val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
        val model = ONNXModels.PoseDetection.MoveNetSinglePoseLighting.pretrainedModel(modelHub)

        model.use { poseDetectionModel ->
            val imageFile = getFileFromResource("datasets/poses/single/1.jpg")
            val detectedPose = poseDetectionModel.detectPose(imageFile = imageFile)
            assertEquals(17, detectedPose.landmarks.size)
            assertEquals(18, detectedPose.edges.size)
        }
    }

    @Test
    fun easyPoseDetectionMoveNetMultiPoseLightingTest() {
        val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
        val model = ONNXModels.PoseDetection.MoveNetMultiPoseLighting.pretrainedModel(modelHub)

        model.use { poseDetectionModel ->
            val imageFile = getFileFromResource("datasets/poses/multi/1.jpg")
            val detectedPoses = poseDetectionModel.detectPoses(imageFile = imageFile)
            assertEquals(3, detectedPoses.poses.size)
            detectedPoses.poses.forEach {
                assertEquals(17, it.second.landmarks.size)
                assertEquals(18, it.second.edges.size)
            }
        }

    }

    @Test
    fun poseDetectionMoveNetSinglePoseLightingTest() {
        val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
        val modelType = ONNXModels.PoseDetection.MoveNetSinglePoseLighting
        val model = modelHub.loadModel(modelType)

        model.use {
            val imageFile = getFileFromResource("datasets/poses/single/1.jpg")
            val fileDataLoader = pipeline<BufferedImage>()
                .resize {
                    outputHeight = 192
                    outputWidth = 192
                }
                .convert { colorMode = ColorMode.BGR }
                .toFloatArray { }
                .call(modelType.preprocessor)
                .fileLoader()

            val inputData = fileDataLoader.load(imageFile)

            val rawPoseLandMarks = it.predictRaw(inputData) { result ->
                result.get2DFloatArray("output_0")
            }

            assertEquals(17, rawPoseLandMarks.size)
        }
    }

    @Test
    fun poseDetectionMoveNetSinglePoseThunderTest() {
        val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
        val modelType = ONNXModels.PoseDetection.MoveNetSinglePoseThunder
        val model = modelHub.loadModel(modelType)

        model.use {
            val imageFile = getFileFromResource("datasets/poses/single/1.jpg")

            val preprocessing = pipeline<BufferedImage>()
                .resize {
                    outputHeight = 256
                    outputWidth = 256
                }
                .convert { colorMode = ColorMode.BGR }
                .toFloatArray { }
                .call(modelType.preprocessor)
                .fileLoader()

            val inputData = preprocessing.load(imageFile)

            val rawPoseLandMarks = it.predictRaw(inputData) { result ->
                result.get2DFloatArray("output_0")
            }

            assertEquals(17, rawPoseLandMarks.size)
        }
    }

    @Test
    fun poseDetectionMoveNetMultiPoseLightingTest() {
        val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
        val modelType = ONNXModels.PoseDetection.MoveNetMultiPoseLighting
        val model = modelHub.loadModel(modelType)

        model.use { inferenceModel ->
            val imageFile = getFileFromResource("datasets/poses/multi/1.jpg")

            val dataLoader = pipeline<BufferedImage>()
                .resize {
                    outputHeight = 256
                    outputWidth = 256
                }
                .convert { colorMode = ColorMode.BGR }
                .toFloatArray { }
                .call(modelType.preprocessor)
                .fileLoader()


            val inputData = dataLoader.load(imageFile)
            val rawPosesLandMarks = inferenceModel.predictRaw(inputData) { result ->
                result.get2DFloatArray("output_0")
            }
            println(rawPosesLandMarks.contentDeepToString())

            assertEquals(6, rawPosesLandMarks.size)
            rawPosesLandMarks.forEach {
                assertEquals(56, it.size)
            }
        }
    }
}
