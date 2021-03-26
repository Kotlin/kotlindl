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
    public lateinit var imagePreprocessing: ImagePreprocessing
    public lateinit var rescaling: Rescaling

    public fun handleFile(file: File): FloatArray {
        //TODO: call stage if initialized and used, should be implemented an empty stage, which returns just Image or just the same floatArray

        var image = imagePreprocessing.load.fileToImage(file)
        var shape = imagePreprocessing.load.imageShape

        if (imagePreprocessing.isCropInitialized) {
            val (croppedImage, croppedShape) = imagePreprocessing.crop.apply(image, shape)
            image = croppedImage
            shape = croppedShape
        }

        if (imagePreprocessing.isRotateInitialized) {
            val (rotatedImage, rotatedShape) = imagePreprocessing.rotate.apply(image, shape)
            image = rotatedImage
            shape = rotatedShape
        }

        if (imagePreprocessing.isResizeInitialized) {
            val (resizedImage, resizedShape) = imagePreprocessing.resize.apply(image, shape)
            image = resizedImage
            shape = resizedShape
        }

        val floatArray = OnHeapDataset.toRawVector(
            imageToByteArray(image, imagePreprocessing.load.colorMode)
        )

        return if (::rescaling.isInitialized) {
            rescaling.apply(floatArray)
        } else floatArray
    }
}

public fun preprocessing(init: Preprocessing.() -> Unit): Preprocessing =
    Preprocessing()
        .apply(init)

public fun Preprocessing.imagePreprocessing(block: ImagePreprocessing.() -> Unit) {
    imagePreprocessing = ImagePreprocessing().apply(block)
}

public fun Preprocessing.rescale(block: Rescaling.() -> Unit) {
    rescaling = Rescaling().apply(block)
}





