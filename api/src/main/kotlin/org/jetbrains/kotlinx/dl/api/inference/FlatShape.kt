/*
 * Copyright 2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference

/**
 * Represents a 2D geometric shape.
 */
public interface FlatShape<T : FlatShape<T>> {
    /**
     * Creates a new geometric shape of the same type by applying the provided [mapping] to the coordinates of the current shape.
     */
    public fun map(mapping: (Float, Float) -> Pair<Float, Float>): T
}