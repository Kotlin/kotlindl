/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.facealignment

import org.jetbrains.kotlinx.dl.api.inference.FlatShape

/**
 * Represents a face landmark as a point on the image with two coordinates relative to the top-left corner.
 * Both coordinates have values between 0 and 1.
 * */
public data class Landmark(public val x: Float, public val y: Float) : FlatShape<Landmark> {
    override fun map(mapping: (Float, Float) -> Pair<Float, Float>): Landmark {
        val (x1, y1) = mapping(x, y)
        return Landmark(x1, y1)
    }
}
