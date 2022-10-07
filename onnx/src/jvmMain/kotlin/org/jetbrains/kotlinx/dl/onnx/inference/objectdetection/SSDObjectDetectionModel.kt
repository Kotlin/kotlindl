/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.onnx.inference.objectdetection

import ai.onnxruntime.OrtSession
import org.jetbrains.kotlinx.dl.api.core.FloatData
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.preprocessing.Operation
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.impl.dataset.Coco
import org.jetbrains.kotlinx.dl.impl.preprocessing.call
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.*
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModels
import org.jetbrains.kotlinx.dl.onnx.inference.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.onnx.inference.OrtSessionResultConversions.get2DFloatArray
import org.jetbrains.kotlinx.dl.onnx.inference.OrtSessionResultConversions.getFloatArray
import org.jetbrains.kotlinx.dl.onnx.inference.OrtSessionResultConversions.getLongArray
import java.awt.image.BufferedImage
import java.io.File
import java.io.IOException


private val SSD_RESNET_METADATA = SSDLikeModelMetadata("bboxes", "labels", "scores", 1, 0)

/**
 * Special model class for detection objects on images
 * with built-in preprocessing and post-processing.
 *
 * It internally uses [ONNXModels.ObjectDetection.SSD] trained on the COCO dataset.
 *
 * Note that output class labels do not correspond to ids in COCO annotations.
 * If you want to evaluate this model on the COCO validation/test set, you need to convert class predictions using appropriate mapping.
 *
 *
 *  @see <a href="https://github.com/lji72/inference/blob/tf_ssd_resent34_align_onnx/others/cloud/single_stage_detector/tensorflow/dataset_config/coco_labelmap.txt">
 *     Example mapping</a>
 *
 * @param [internalModel] model used to make predictions

 * @since 0.3
 */
public class SSDObjectDetectionModel(
    override val internalModel: OnnxInferenceModel,
    modelKindDescription: String? = null
) : SSDLikeModelBase<BufferedImage>(SSD_RESNET_METADATA, modelKindDescription), InferenceModel by internalModel {

    override val preprocessing: Operation<BufferedImage, FloatData>
        get() = pipeline<BufferedImage>()
            .resize {
                outputHeight = inputDimensions[1].toInt()
                outputWidth = inputDimensions[2].toInt()
            }
            .convert { colorMode = ColorMode.RGB }
            .toFloatArray { }
            .call(ONNXModels.ObjectDetection.SSD.preprocessor)

    override val classLabels: Map<Int, String> = Coco.V2014.labels()

    /**
     * Constructs the object detection model from a given path.
     * @param [pathToModel] path to model
     */
    public constructor(pathToModel: String) : this(OnnxInferenceModel(pathToModel))

    /**
     * Returns the top N detected object for the given image file sorted by the score.
     *
     * NOTE: this method includes the SSD - related preprocessing.
     *
     * @param [imageFile] File, should be an image.
     * @param [topK] The number of the detected objects with the highest score to be returned.
     * @return List of [DetectedObject] sorted by score.
     */
    @Throws(IOException::class)
    public fun detectObjects(imageFile: File, topK: Int = 5): List<DetectedObject> {
        return detectObjects(ImageConverter.toBufferedImage(imageFile), topK)
    }

    // TODO remove code duplication due to different type of class labels array
    override fun convert(output: OrtSession.Result): List<DetectedObject> {
        val boxes = output.get2DFloatArray(metadata.outputBoxesName)
        val classIndices = output.getLongArray(metadata.outputClassesName)
        val probabilities = output.getFloatArray(metadata.outputScoresName)
        val numberOfFoundObjects = boxes.size

        val foundObjects = mutableListOf<DetectedObject>()
        for (i in 0 until numberOfFoundObjects) {
            val detectedObject = DetectedObject(
                xMin = boxes[i][metadata.xMinIdx],
                xMax = boxes[i][metadata.xMinIdx + 2],
                // left, bot, right, top
                yMin = boxes[i][metadata.yMinIdx],
                yMax = boxes[i][metadata.yMinIdx + 2],
                probability = probabilities[i],
                label = classLabels[classIndices[i].toInt()]
            )
            foundObjects.add(detectedObject)
        }
        return foundObjects
    }


    override fun copy(
        copiedModelName: String?,
        saveOptimizerState: Boolean,
        copyWeights: Boolean
    ): SSDObjectDetectionModel {
        return SSDObjectDetectionModel(
            internalModel.copy(copiedModelName, saveOptimizerState, copyWeights),
            modelKindDescription
        )
    }
}
