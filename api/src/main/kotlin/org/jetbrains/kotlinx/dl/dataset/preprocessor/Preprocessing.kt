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

    public operator fun invoke(): Pair<FloatArray, ImageShape> {
        val file = imagePreprocessingStage.load.pathToData
        require(file!!.isFile) { "Invoke call is available for one file preprocessing only." }

        return handleFile(file)
    }

    internal fun handleFile(file: File): Pair<FloatArray, ImageShape> {
        //TODO: call stage if initialized and used, should be implemented an empty stage, which returns just Image or just the same floatArray
        val filename = file.name
        var image = imagePreprocessingStage.load.fileToImage(file)
        var shape = imagePreprocessingStage.load.imageShape!!

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

        val floatArray = OnHeapDataset.toRawVector(
            imageToByteArray(image, imagePreprocessingStage.load.colorMode)
        )

        return if (::rescalingStage.isInitialized) {
            Pair(rescalingStage.apply(floatArray), shape)
        } else Pair(floatArray, shape)
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
