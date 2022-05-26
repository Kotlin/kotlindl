/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor

import org.jetbrains.kotlinx.dl.dataset.image.ImageConverter
import org.jetbrains.kotlinx.dl.dataset.image.getShape
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.ImagePreprocessing
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.ImagePreprocessorBase
import java.awt.image.BufferedImage
import java.io.File

/**
 * The data preprocessing pipeline presented as Kotlin DSL on receivers.
 *
 * Could be used to handle directory of images or one image file.
 */
public class Preprocessing {
    /** This stage describes the process of image loading and transformation before converting to tensor. */
    public lateinit var imagePreprocessingStage: ImagePreprocessing

    /** This stage describes the process of data transformation after converting to tensor. */
    public lateinit var tensorPreprocessingStage: TensorPreprocessing

    /**
     * Returns the final shape of data when image preprocessing is applied to the image with the given shape.
     * @param [inputShape] shape of the input image
     * @return shape of the output image
     * */
    public fun getFinalShape(inputShape: ImageShape = ImageShape()): ImageShape {
        var imageShape = inputShape
        if (::imagePreprocessingStage.isInitialized) {
            for (operation in imagePreprocessingStage.operations) {
                imageShape = operation.getOutputShape(imageShape)
            }
        }
        if (imageShape.width == null && imageShape.height == null && imageShape.channels == null) {
            throw IllegalStateException(
                "Final image shape is unclear. Operator with fixed output size (such as \"resize\") should be used " +
                        "or ImageShape with height, weight and channels should be passed as a parameter."
            )
        }
        return imageShape
    }

    /** Applies the preprocessing pipeline to the specific image file. */
    public operator fun invoke(imageFile: File): Pair<FloatArray, ImageShape> {
        require(imageFile.exists()) { "File '$imageFile' does not exist." }
        require(imageFile.isFile) {
            if (imageFile.isDirectory) "File '$imageFile' is a directory."
            else "File '$imageFile' is not a normal file."
        }
        return handleFile(imageFile)
    }

    internal fun handleFile(file: File): Pair<FloatArray, ImageShape> {
        val image = file.inputStream().use { inputStream -> ImageConverter.toBufferedImage(inputStream) }
        return handleImage(image, file.name)
    }

    internal fun handleImage(inputImage: BufferedImage, imageName: String): Pair<FloatArray, ImageShape> {
        var image = inputImage
        if (::imagePreprocessingStage.isInitialized) {
            for (operation in imagePreprocessingStage.operations) {
                image = operation.apply(image)
                (operation as? ImagePreprocessorBase)?.save?.save(imageName, image)
            }
        }

        var tensor = ImageConverter.toRawFloatArray(image)
        val shape = image.getShape()

        if (::tensorPreprocessingStage.isInitialized) {
            for (operation in tensorPreprocessingStage.operations) {
                tensor = operation.apply(tensor, shape)
            }
        }

        return Pair(tensor, shape)
    }
}

/** Defines preprocessing operations. */
public fun preprocess(init: Preprocessing.() -> Unit): Preprocessing =
    Preprocessing()
        .apply(init)

/** Defines preprocessing operations on the image. */
public fun Preprocessing.transformImage(block: ImagePreprocessing.() -> Unit) {
    imagePreprocessingStage = ImagePreprocessing().apply(block)
}

/** Defines preprocessing operations on the tensor. */
public fun Preprocessing.transformTensor(block: TensorPreprocessing.() -> Unit) {
    tensorPreprocessingStage = TensorPreprocessing().apply(block)
}