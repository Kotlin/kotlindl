/*
 * Copyright 2020-2023 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.summary

/**
 * Interface for the models for which we can produce meaningful summary.
 */
public interface ModelWithSummary {
    /**
     * Returns model summary.
     *
     * @return model summary
     */
    public fun summary(): ModelSummary
}
