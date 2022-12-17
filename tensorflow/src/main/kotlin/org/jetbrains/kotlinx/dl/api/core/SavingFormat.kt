/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core

/** Model saving format. */
public sealed class SavingFormat {
    /** Saves model as graph in .pb file 1.15 Tensorflow format and variables in .txt file format. */
    public object TfGraphCustomVariables : SavingFormat()

    /** Saves model as graph in .pb file 1.15 Tensorflow format without variables data. */
    public object TfGraph : SavingFormat()

    /**
     * Saves model as a list of layers in .json file format and variables in .txt file format.
     * @property [isKerasFullyCompatible] If true, it generates fully Keras-compatible configuration.
     * */
    public class JsonConfigCustomVariables(public val isKerasFullyCompatible: Boolean = false) : SavingFormat()
}
