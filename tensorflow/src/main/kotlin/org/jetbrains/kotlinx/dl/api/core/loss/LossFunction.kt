/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.loss

import org.tensorflow.Operand
import org.tensorflow.op.Ops

/**
 * Basic interface for all loss functions.
 */
public abstract class LossFunction(
    /** Reduction type. Should be defined in subclass*/
    public val reductionType: ReductionType
) {
    /**
     * Applies [LossFunction] to the [yPred] labels predicted by the model and known [yTrue] hidden during training.
     *
     * @param yPred The predicted values. shape = `[batch_size, d0, .. dN]`.
     * @param yTrue Ground truth values. Shape = `[batch_size, d0, .. dN]`, except
     * sparse loss functions such as sparse categorical crossentropy where
     * shape = `[batch_size, d0, .. dN-1]`.
     * @param [tf] TensorFlow graph API for building operations.
     */
    public abstract fun apply(
        tf: Ops,
        yPred: Operand<Float>,
        yTrue: Operand<Float>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float>
}

