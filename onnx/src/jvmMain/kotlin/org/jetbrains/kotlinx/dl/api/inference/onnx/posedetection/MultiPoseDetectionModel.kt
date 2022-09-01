/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.onnx.posedetection

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.api.inference.posedetection.DetectedPose
import org.jetbrains.kotlinx.dl.api.inference.posedetection.MultiPoseDetectionResult
import org.jetbrains.kotlinx.dl.api.inference.posedetection.PoseLandmark
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.image.ImageConverter
import org.jetbrains.kotlinx.dl.dataset.preprocessing.Operation
import org.jetbrains.kotlinx.dl.dataset.preprocessing.call
import org.jetbrains.kotlinx.dl.dataset.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.toFloatArray
import java.awt.image.BufferedImage
import java.io.File
import java.io.IOException

private const val CLASS_LABEL = "person"
private const val OUTPUT_NAME = "output_0"
private const val INPUT_SIZE = 256

/**
 * MultiPoseDetectionModel is an ultra-fast and accurate model that detects 6 persons with 17 keypoints and 18 basic edges of a body for each of them.
 *
 * It internally uses [ONNXModels.PoseDetection.MoveNetMultiPoseLighting] under the hood to make predictions.
 *
 * @param [internalModel] model used to make predictions
 */
public class MultiPoseDetectionModel(private val internalModel: OnnxInferenceModel) : InferenceModel by internalModel {
    private val preprocessing: Operation<BufferedImage, Pair<FloatArray, TensorShape>>
        get() = pipeline<BufferedImage>()
            .resize {
                outputHeight = INPUT_SIZE
                outputWidth = INPUT_SIZE
            }
            .convert { colorMode = ColorMode.RGB }
            .toFloatArray { }
            .call(ONNXModels.PoseDetection.MoveNetSinglePoseLighting.preprocessor)

    /**
     * Constructs the pose detection model from a given path.
     * @param [pathToModel] path to model
     */
    public constructor(pathToModel: String): this(OnnxInferenceModel(pathToModel))

    public fun detectPoses(inputData: FloatArray, confidence: Float = 0.005f): MultiPoseDetectionResult {
        val rawPrediction = internalModel.predictRaw(inputData)
        val rawPoseLandMarks = (rawPrediction[OUTPUT_NAME] as Array<Array<FloatArray>>)[0]

        val poses = mutableListOf<Pair<DetectedObject, DetectedPose>>()

        rawPoseLandMarks.forEachIndexed { poseIndex, floats ->
            val foundPoseLandmarks = mutableListOf<PoseLandmark>()

            for (keyPointIdx in 0..16) {
                val poseLandmark = PoseLandmark(
                    poseLandmarkLabel = keyPoints[keyPointIdx]!!,
                    x = floats[3 * keyPointIdx + 1],
                    y = floats[3 * keyPointIdx],
                    probability = floats[3 * keyPointIdx + 2]
                )
                foundPoseLandmarks.add(poseLandmark)
            }

            // [ymin, xmin, ymax, xmax, score]
            val detectedObject = DetectedObject(
                classLabel = CLASS_LABEL,
                probability = floats[55],
                yMin = floats[53],
                xMin = floats[52],
                yMax = floats[51],
                xMax = floats[54]
            )

            val foundPoseEdges = buildPoseEdges(foundPoseLandmarks)
            val detectedPose = DetectedPose(foundPoseLandmarks, foundPoseEdges)

            if (detectedObject.probability > confidence) poses.add(
                Pair(detectedObject, detectedPose)
            )
        }

        return MultiPoseDetectionResult(poses)
    }

    public fun detectPoses(image: BufferedImage, confidence: Float = 0.1f): MultiPoseDetectionResult {
        return detectPoses(preprocessing.apply(image).first, confidence)
    }

    @Throws(IOException::class)
    public fun detectPoses(imageFile: File, confidence: Float = 0.1f): MultiPoseDetectionResult {
        return detectPoses(ImageConverter.toBufferedImage(imageFile), confidence)
    }

    override fun copy(
        copiedModelName: String?,
        saveOptimizerState: Boolean,
        copyWeights: Boolean
    ): MultiPoseDetectionModel {
        return MultiPoseDetectionModel(internalModel.copy(copiedModelName, saveOptimizerState, copyWeights))
    }
}
