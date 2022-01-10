/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor.generator

import java.io.File

/**
 * This [LabelGenerator] is responsible for creation default labels with value Float.NaN.
 */
public class EmptyLabels : LabelGenerator {
    override fun getLabel(file: File): Float = Float.NaN
}
