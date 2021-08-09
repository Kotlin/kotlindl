/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor.generator

/**
 * This [LabelGenerator] is responsible for extracting labels from the names of folders where the images of the appropriate class are located.
 * It keeps the [mapping] name of classes to int numbers.
 *
 * @property [mapping] The mapping from class names to class labels presented as natural numbers.
 */
public class FromFolders(public val mapping: Map<String, Int>) : LabelGenerator
