/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.regularizer

import org.tensorflow.Operand
import org.tensorflow.op.Ops

/**
 * Regularizers allow you to apply penalties on layer parameters or layer
 * activity during optimization. These penalties are summed into the loss
 * function that the network optimizes.
 *
 * Regularization penalties are applied on a per-layer basis.
 * The exact API will depend on the layer, but many layers (e.g. `Dense`, `Conv1D`, `Conv2D` and `Conv3D`) have a unified API.
 * These layers expose 3 keyword arguments:
 * - `kernel_regularizer`: Regularizer to apply a penalty on the layer's kernel
 * - `bias_regularizer`: Regularizer to apply a penalty on the layer's bias
 * - `activity_regularizer`: Regularizer to apply a penalty on the layer's output
 *
 * NOTE: The value returned by the `activity_regularizer` is divided by the input
 * batch size so that the relative weighting between the weight regularizers and
 * the activity regularizers does not change with the batch size.
 */
public abstract class Regularizer {
    /** Applies regularization to the input. */
    public abstract fun apply(
        tf: Ops,
        input: Operand<Float>,
    ): Operand<Float>
}
