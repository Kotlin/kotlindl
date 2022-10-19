/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.exception

/**
 * Thrown by [org.jetbrains.kotlinx.dl.api.core.Sequential] model during model initialization if the model layers has the same name in a few layers.
 */
public class RepeatableLayerNameException(layerName: String) :
    Exception("The layer name $layerName is used in previous layers. The layer name should be unique.")
