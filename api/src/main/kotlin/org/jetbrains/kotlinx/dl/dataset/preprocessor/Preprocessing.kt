/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor

import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import org.jetbrains.kotlinx.dl.dataset.image.ImageConverter.Companion.imageToByteArray
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.ImagePreprocessing
import java.io.File

/**
 * The data preprocessing pipeline presented as Kotlin DSL on receivers.
 *
 * Could be used to handle directory of images or one image file.
 */
public class Preprocessing {
    public lateinit var imagePreprocessingStage: ImagePreprocessing

    public lateinit var tensorPreprocessingStage: TensorPreprocessing

    // TODO: rewrite correctly with all stages not only loading and resize
    public val finalShape: ImageShape
        get() {
            return if (imagePreprocessingStage.isResizeInitialized && imagePreprocessingStage.isLoadInitialized) {
                ImageShape(
                    imagePreprocessingStage.resize.outputWidth.toLong(),
                    imagePreprocessingStage.resize.outputHeight.toLong(),
                    imagePreprocessingStage.load.imageShape!!.channels
                )
            } else if (imagePreprocessingStage.load.imageShape!!.width != null && imagePreprocessingStage.load.imageShape!!.height != null) {
                ImageShape(
                    imagePreprocessingStage.load.imageShape!!.width,
                    imagePreprocessingStage.load.imageShape!!.height,
                    imagePreprocessingStage.load.imageShape!!.channels
                )
            } else {
                throw IllegalStateException("Final image shape is unclear. The resize operator should be initialized or imageShape with height, weight and channels should be initialized.")
            }
        }

    /** Preprocessing one image file via described preprocessing pipeline. */
    public operator fun invoke(): Pair<FloatArray, ImageShape> {
        val file = imagePreprocessingStage.load.pathToData
        require(file!!.isFile) { "Invoke call is available for one file preprocessing only." }

        return handleFile(file)
    }

    // TODO: need a method a-la outputShape after preprocessing
    internal fun handleFile(file: File): Pair<FloatArray, ImageShape> {
        //TODO: call stage if initialized and used, should be implemented an empty stage, which returns just Image or just the same floatArray
        val filename = file.name
        var image = imagePreprocessingStage.load.fileToImage(file)
        var shape = imagePreprocessingStage.load.imageShape!!
        // TODO: handle if height and width are missed in imageShape in load stage
        // if both nulls write a warning to logs about possible mismatch
        if (imagePreprocessingStage.isCropInitialized) {
            val (croppedImage, croppedShape) = imagePreprocessingStage.crop.apply(image, shape)
            image = croppedImage
            shape = croppedShape
        }

        if (imagePreprocessingStage.isRotateInitialized) {
            val (rotatedImage, rotatedShape) = imagePreprocessingStage.rotate.apply(image, shape)
            image = rotatedImage
            shape = rotatedShape
        }

        if (imagePreprocessingStage.isResizeInitialized) {
            val (resizedImage, resizedShape) = imagePreprocessingStage.resize.apply(image, shape)
            image = resizedImage
            shape = resizedShape
        }

        if (imagePreprocessingStage.isSaveInitialized) {
            imagePreprocessingStage.save.imageToFile(filename, image, shape)
        }

        var tensor = OnHeapDataset.toRawVector(
            imageToByteArray(image, imagePreprocessingStage.load.colorMode)
        )

        if (::tensorPreprocessingStage.isInitialized) {
            if (tensorPreprocessingStage.isRescalingInitialized)
                tensor = tensorPreprocessingStage.rescaling.apply(tensor, shape)

            if (tensorPreprocessingStage.isSharpenInitialized)
                tensor = tensorPreprocessingStage.sharpen.apply(tensor, shape)
        }

        return Pair(tensor, shape)
    }
}

/** */
public fun preprocess(init: Preprocessing.() -> Unit): Preprocessing =
    Preprocessing()
        .apply(init)

/** */
public fun Preprocessing.transformImage(block: ImagePreprocessing.() -> Unit) {
    imagePreprocessingStage = ImagePreprocessing().apply(block)
}

/** */
public fun Preprocessing.transformTensor(block: TensorPreprocessing.() -> Unit) {
    tensorPreprocessingStage = TensorPreprocessing().apply(block)
}
