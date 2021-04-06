/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor

import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import org.jetbrains.kotlinx.dl.dataset.image.ImageConverter.Companion.imageToByteArray
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.ImagePreprocessing
import java.io.File

public class Preprocessing {
    // TODO: maybe it should be list of imageStages https://proandroiddev.com/writing-dsls-in-kotlin-part-2-cd9dcd0c4715 для целей расширения DSL кастомными классами
    public lateinit var imagePreprocessingStage: ImagePreprocessing

    public lateinit var rescalingStage: Rescaling

    public lateinit var sharpeningStage: Sharpen

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
            } // TODO: add test for this cases
        }

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

        if (::rescalingStage.isInitialized)
            tensor = rescalingStage.apply(tensor, shape)

        if (::sharpeningStage.isInitialized)
            tensor = sharpeningStage.apply(tensor, shape)

        return Pair(tensor, shape)
    }
}

public fun preprocessingPipeline(init: Preprocessing.() -> Unit): Preprocessing =
    Preprocessing()
        .apply(init)

public fun Preprocessing.imagePreprocessing(block: ImagePreprocessing.() -> Unit) {
    imagePreprocessingStage = ImagePreprocessing().apply(block)
}

public fun Preprocessing.rescale(block: Rescaling.() -> Unit) {
    rescalingStage = Rescaling().apply(block)
}

public fun Preprocessing.sharpen(block: Sharpen.() -> Unit) {
    sharpeningStage = Sharpen().apply(block)
}
