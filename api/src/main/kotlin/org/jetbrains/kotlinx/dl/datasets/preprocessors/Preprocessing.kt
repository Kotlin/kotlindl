/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.datasets.preprocessors

import org.jetbrains.kotlinx.dl.datasets.OnHeapDataset
import org.jetbrains.kotlinx.dl.datasets.image.ImageConverter.Companion.imageToByteArray
import org.jetbrains.kotlinx.dl.datasets.preprocessors.image.ImagePreprocessing
import java.io.File

public class Preprocessing {
    // TODO: maybe it should be list of imageStages https://proandroiddev.com/writing-dsls-in-kotlin-part-2-cd9dcd0c4715 для целей расширения DSL кастомными классами
    public lateinit var imagePreprocessingStage: ImagePreprocessing

    public lateinit var rescalingStage: Rescaling

    public fun handleFile(file: File): FloatArray {
        //TODO: call stage if initialized and used, should be implemented an empty stage, which returns just Image or just the same floatArray

        var image = imagePreprocessingStage.load.fileToImage(file)
        var shape = imagePreprocessingStage.load.imageShape

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

        val floatArray = OnHeapDataset.toRawVector(
            imageToByteArray(image, imagePreprocessingStage.load.colorMode)
        )

        return if (::rescalingStage.isInitialized) {
            rescalingStage.apply(floatArray)
        } else floatArray
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
