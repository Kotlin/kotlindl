/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core

/** Model saving format. */
public enum class SavingFormat {
    /** Saves model as graph in .pb file 1.15 Tensorflow format and variables in .txt file format. */
    TF_GRAPH_CUSTOM_VARIABLES,

    /** Saves model as graph in .pb file 1.15 Tensorflow format without variables data. */
    TF_GRAPH,

    /** Saves model as a list of layers in .json file format and variables in .txt file format. */
    JSON_CONFIG_CUSTOM_VARIABLES
}
