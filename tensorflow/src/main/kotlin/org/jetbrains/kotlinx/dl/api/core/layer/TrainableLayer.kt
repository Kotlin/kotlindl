/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

/**
 * Represents a [ParametrizedLayer] that can be trained.
 */
public interface TrainableLayer : ParametrizedLayer {
    /**
     * True, if layer's weights could be changed during training.
     * If false, layer's weights are frozen and could not be changed during training.
     */
    public var isTrainable: Boolean
}

/**
 * Returns true if this [Layer] is trainable, false otherwise.
 * Always returns false for layers not implementing [TrainableLayer].
 */
public val Layer.isTrainable: Boolean
    get() = if (this is TrainableLayer) isTrainable else false

/**
 * Freezes layer weights, so they won't be changed during training.
 */
public fun Layer.freeze() {
    if (this is TrainableLayer) isTrainable = false
}

/**
 * Unfreezes layer weights, allowing to change them during training.
 */
public fun Layer.unfreeze() {
    require(this is TrainableLayer) {
        "Layer ${this.name} does not support training"
    }
    isTrainable = true
}