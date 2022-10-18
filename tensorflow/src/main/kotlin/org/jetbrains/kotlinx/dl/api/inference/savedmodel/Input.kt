/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.savedmodel

/**
 * Possible inputs for static TensorFlow graph in [SavedModel].
 *
 * @property [tfName] Maps TensorFlow operand name to enum value.
 */
public enum class Input(public val tfName: String) {
    /** Placeholder. */
    PLACEHOLDER("Placeholder")
}
