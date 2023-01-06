/*
 * Copyright 2020-2023 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.inference.imagerecognition

import org.jetbrains.kotlinx.dl.api.core.FloatData
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.preprocessing.Identity
import org.jetbrains.kotlinx.dl.api.preprocessing.Operation
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.impl.dataset.Imagenet
import org.jetbrains.kotlinx.dl.impl.preprocessing.call
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.*
import java.awt.image.BufferedImage
import java.io.File
import java.io.IOException

/**
 * The light-weight API for solving Image Recognition task with one of the Model Hub models trained on ImageNet dataset.
 */
public class ImageRecognitionModel(
    internalModel: InferenceModel,
    private val inputColorMode: ColorMode,
    private val channelsFirst: Boolean,
    private val preprocessor: Operation<FloatData, FloatData> = Identity(),
    modelKindDescription: String? = null
) : ImageRecognitionModelBase<BufferedImage>(internalModel, modelKindDescription) {
    /** Class labels for ImageNet dataset. */
    override val classLabels: Map<Int, String> = Imagenet.V1k.labels()

    override val preprocessing: Operation<BufferedImage, FloatData>
        get() = createPreprocessing(internalModel, channelsFirst, inputColorMode, preprocessor)

    /**
     * Predicts [topK] objects for the given [imageFile].
     * Default preprocessing [Operation] is applied to an image.
     *
     * @param [imageFile] Input image [File].
     * @param [topK] Number of top ranked predictions to return
     *
     * @see preprocessing
     *
     * @return The list of pairs <label, probability> sorted from the most probable to the lowest probable.
     */
    @Throws(IOException::class)
    public fun predictTopKObjects(imageFile: File, topK: Int): List<Pair<String, Float>> {
        return predictTopKObjects(ImageConverter.toBufferedImage(imageFile), topK)
    }

    /**
     * Predicts object for the given [imageFile].
     * Default preprocessing [Operation] is applied to an image.
     *
     * @param [imageFile] Input image [File].
     * @see preprocessing
     *
     * @return The label of the recognized object with the highest probability.
     */
    @Throws(IOException::class)
    public fun predictObject(imageFile: File): String {
        return predictObject(ImageConverter.toBufferedImage(imageFile))
    }

    public companion object {
        /**
         * Creates a preprocessing [Operation] which converts given [BufferedImage] to [FloatData].
         * Image is resized to fit the [model] input dimensions (according to the [channelsFirst] property),
         * converted to the given [inputColorMode], transformed to the [FloatArray] which is processed with the given
         * [preprocessor].
         */
        public fun createPreprocessing(model: InferenceModel,
                                       channelsFirst: Boolean,
                                       inputColorMode: ColorMode,
                                       preprocessor: Operation<FloatData, FloatData> = Identity()
        ): Operation<BufferedImage, FloatData> {
            val inputDimensions = model.inputDimensions
            require(inputDimensions.all { it > 0 }) {
                "Model input dimensions are not defined. Current input dimensions: ${inputDimensions.contentToString()}"
            }
            return createPreprocessing(inputDimensions, channelsFirst, inputColorMode, preprocessor)
        }

        /**
         * Creates a preprocessing [Operation] which converts given [BufferedImage] to [FloatData].
         * Image is resized to fit the [inputDimensions] (according to the [channelsFirst] property),
         * converted to the given [inputColorMode], transformed to the [FloatArray] which is processed with the given
         * [preprocessor].
         */
        public fun createPreprocessing(inputDimensions: LongArray,
                                       channelsFirst: Boolean,
                                       inputColorMode: ColorMode,
                                       preprocessor: Operation<FloatData, FloatData> = Identity()
        ): Operation<BufferedImage, FloatData> {
            val (width, height) = if (channelsFirst)
                Pair(inputDimensions[1], inputDimensions[2])
            else
                Pair(inputDimensions[0], inputDimensions[1])
            return pipeline<BufferedImage>()
                .resize {
                    outputHeight = height.toInt()
                    outputWidth = width.toInt()
                    interpolation = InterpolationType.BILINEAR
                }
                .convert { colorMode = inputColorMode }
                .toFloatArray {}
                .call(preprocessor)
        }
    }
}
