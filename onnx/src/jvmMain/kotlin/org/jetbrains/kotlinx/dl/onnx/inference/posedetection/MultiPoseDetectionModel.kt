/*
 * Copyright 2020-2023 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.onnx.inference.posedetection

import org.jetbrains.kotlinx.dl.api.core.FloatData
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.posedetection.MultiPoseDetectionResult
import org.jetbrains.kotlinx.dl.api.preprocessing.Operation
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.impl.preprocessing.call
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.*
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModels
import org.jetbrains.kotlinx.dl.onnx.inference.OnnxInferenceModel
import java.awt.image.BufferedImage
import java.io.File

private const val OUTPUT_NAME = "output_0"

/**
 * MultiPoseDetectionModel is an ultra-fast and accurate model that detects 6 persons with 17 keypoints and 18 basic edges of a body for each of them.
 *
 * It internally uses [ONNXModels.PoseDetection.MoveNetMultiPoseLighting] under the hood to make predictions.
 *
 * @param [internalModel] model used to make predictions
 */
public class MultiPoseDetectionModel(
    override val internalModel: OnnxInferenceModel,
    private var inputShape: LongArray,
    modelKindDescription: String? = null
) : MultiPoseDetectionModelBase<BufferedImage>(modelKindDescription) {

    override val preprocessing: Operation<BufferedImage, FloatData>
        get() = pipeline<BufferedImage>()
            .resize {
                outputHeight = inputShape[0].toInt()
                outputWidth = inputShape[1].toInt()
            }
            .convert { colorMode = ColorMode.RGB }
            .toFloatArray { }
            .call(ONNXModels.PoseDetection.MoveNetSinglePoseLighting.preprocessor)

    override val outputName: String = OUTPUT_NAME
    override val keyPointsLabels: Map<Int, String> = keyPoints
    override val edgeKeyPoints: List<Pair<Int, Int>> = edgeKeyPointsPairs

    /**
     * Constructs the pose detection model from a given path.
     * @param [pathToModel] path to model
     */
    public constructor(pathToModel: String, inputShape: LongArray) : this(OnnxInferenceModel(pathToModel), inputShape)

    /**
     * Detects poses for the given [imageFile] with the given [confidence].
     * @param [imageFile] file containing an input image
     * @param [confidence] confidence value to use
     */
    public fun detectPoses(imageFile: File, confidence: Float = 0.1f): MultiPoseDetectionResult {
        return detectPoses(ImageConverter.toBufferedImage(imageFile), confidence)
    }

    /**
     * Setter for input shape of the internal model. Images are going to be resized to this shape.
     *
     * @param dims The input shape.
     */
    public fun reshape(vararg dims: Long) {
        inputShape = longArrayOf(*dims)
    }

    override fun close(): Unit = internalModel.close()
}
