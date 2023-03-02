/*
 * Copyright 2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape

/**
 * An alias representing a pair of [FloatArray] data and shape of this data.
 */
public typealias FloatData = Pair<FloatArray, TensorShape>

/**
 * Actual data as a [FloatArray].
 */
public inline val FloatData.floats: FloatArray get() = first

/**
 * Shape of the data.
 */
public inline val FloatData.shape: TensorShape get() = second