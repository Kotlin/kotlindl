/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.metric

/**
 * Metrics.
 */
public enum class Metrics {
    /**
     * Computes the rate of true answers.
     *
     * `metric = sum(y_true == y_pred)`
     */
    ACCURACY,

    /**
     * Computes the mean of absolute difference between labels and predictions.
     *
     * `metric = abs(y_true - y_pred)`
     */
    MAE,

    /**
     * Computes the mean of squares of errors between labels and predictions.
     *
     * `metric = square(y_true - y_pred)`
     */
    MSE,

    /**
     * Computes the mean squared logarithmic error between `y_true` and `y_pred`.
     *
     * `loss = square(log(y_true + 1.) - log(y_pred + 1.))`
     */
    MSLE;
}