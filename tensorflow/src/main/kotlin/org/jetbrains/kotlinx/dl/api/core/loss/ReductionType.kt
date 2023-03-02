/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.loss

/** Choose type of loss values reduction in calculation of loss function value on the specific batch. */
public enum class ReductionType {
    /** Scalar sum of weighted losses. */
    SUM,

    /**
     * Scalar `SUM` divided by number of elements in losses (number of losses).
     * This reduction type is not supported when used with outside of built-in training loops with`compile`/`fit`.
     */
    SUM_OVER_BATCH_SIZE
}
