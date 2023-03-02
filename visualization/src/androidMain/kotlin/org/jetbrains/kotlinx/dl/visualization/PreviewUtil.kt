/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.visualization

import androidx.camera.view.PreviewView
import kotlin.math.max
import kotlin.math.min

/**
 * Defines location and size of the actual preview image relative to the [PreviewView].
 * This information can be used to convert from the image coordinate system to the view coordinate system.
 *
 * @property [x] x-coordinate of the top-left corner of the preview image relative to the [PreviewView] component
 * @property [y] y-coordinate of the top-left corner of the preview image relative to the [PreviewView] component
 * @property [width] width of the preview image
 * @property [height] height of the preview image
 *
 * @see getPreviewImageBounds
 */
data class PreviewImageBounds(val x: Float, val y: Float, val width: Float, val height: Float) {
    fun toViewX(imageX: Float) = imageX * width + x
    fun toViewY(imageY: Float) = imageY * height + y
}

/**
 * Calculate the location of the preview image top-left corner (relative to the component top-left corner)
 * and dimensions, to be used for displaying detected objects, for example with the [DetectorViewBase].
 *
 * When camera preview resolution differs from the dimensions of the [PreviewView] used to display camera input,
 * image is scaled and cropped or padded according to the provided [PreviewView.ScaleType]. Because of this,
 * in order to display detected objects on the [PreviewView], their coordinates need to be converted.
 * This method returns [PreviewImageBounds] object containing the necessary information to preform the conversion
 * from the image coordinate system to the view coordinate system.
 *
 * @param [sourceImageWidth]  width of the image from the camera
 * @param [sourceImageHeight] height of the image from the camera
 * @param [viewWidth] width of the target [PreviewView]
 * @param [viewHeight] height of the target [PreviewView]
 * @param [scaleType] scaling option used in the target [PreviewView]
 *
 * @see <a href="https://developer.android.com/training/camerax/preview#scale-type">Scale type</a>
 */
fun getPreviewImageBounds(
    sourceImageWidth: Int,
    sourceImageHeight: Int,
    viewWidth: Int,
    viewHeight: Int,
    scaleType: PreviewView.ScaleType
): PreviewImageBounds {
    val scale = if (scaleType == PreviewView.ScaleType.FILL_START ||
        scaleType == PreviewView.ScaleType.FILL_END ||
        scaleType == PreviewView.ScaleType.FILL_CENTER
    ) {
        max(viewWidth.toFloat() / sourceImageWidth, viewHeight.toFloat() / sourceImageHeight)
    } else {
        min(viewWidth.toFloat() / sourceImageWidth, viewHeight.toFloat() / sourceImageHeight)
    }
    val previewImageWidth = sourceImageWidth * scale
    val previewImageHeight = sourceImageHeight * scale
    return when (scaleType) {
        PreviewView.ScaleType.FILL_START, PreviewView.ScaleType.FIT_START -> {
            PreviewImageBounds(0f, 0f, previewImageWidth, previewImageHeight)
        }
        PreviewView.ScaleType.FILL_END, PreviewView.ScaleType.FIT_END -> {
            PreviewImageBounds(
                viewWidth - previewImageWidth, viewHeight - previewImageHeight,
                previewImageWidth, previewImageHeight
            )
        }
        else -> {
            PreviewImageBounds(
                viewWidth / 2 - previewImageWidth / 2, viewHeight / 2 - previewImageHeight / 2,
                previewImageWidth, previewImageHeight
            )
        }
    }
}