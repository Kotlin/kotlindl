/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.cnn.dogscats

import org.jetbrains.kotlinx.dl.api.core.model.*
import org.junit.jupiter.api.Disabled
import org.junit.jupiter.api.Test

@Disabled
class CatDogTestSuite {
    @Test
    fun resnet50onCatDogDatasetTest() {
        resnet50onDogsVsCatsDataset()
    }

    @Test
    fun resnet50v2onCatDogDatasetTest() {
        runResNetTraining(::resnet50v2Light)
    }

    @Test
    fun resnet101onCatDogDatasetTest() {
        runResNetTraining(::resnet101Light)
    }

    @Test
    fun resnet101v2onCatDogDatasetTest() {
        runResNetTraining(::resnet101v2Light)
    }

    @Test
    fun resnet152onCatDogDatasetTest() {
        runResNetTraining(::resnet152Light)
    }

    @Test
    fun resnet152v2OnCatDogDatasetTest() {
        runResNetTraining(::resnet152v2Light)
    }
}
