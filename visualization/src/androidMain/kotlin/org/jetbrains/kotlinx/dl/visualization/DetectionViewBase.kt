/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.visualization

import android.content.Context
import android.graphics.Canvas
import android.util.AttributeSet
import android.view.View

/**
 * Base class for [View] implementations which visualize detected results on top of the image preview.
 * Derived classes should implement [drawDetection] method to perform actual drawing.
 */
abstract class DetectorViewBase<T>(context: Context, attrs: AttributeSet) : View(context, attrs) {
    /**
     * Detection result to visualize
     */
    private var _detection: T? = null

    /**
     * Draw given detection result on the [Canvas].
     */
    abstract fun Canvas.drawDetection(detection: T)

    /**
     * Called when a new detection result is set.
     */
    open fun onDetectionSet(detection: T?) = Unit

    /**
     * Set current detection result or null if nothing was detected.
     */
    fun setDetection(detection: T?) {
        synchronized(this) {
            _detection = detection

            onDetectionSet(detection)
            postInvalidate()
        }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        synchronized(this) {
            val detection = _detection
            if (detection != null) {
                canvas.drawDetection(detection)
            }
        }
    }
}