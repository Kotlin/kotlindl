/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.cnn.catdog

import org.junit.jupiter.api.Disabled
import org.junit.jupiter.api.Test

@Disabled
class CatDogTestSuite {
    @Test
    fun resnet50onCatDogDatasetTest() {
        resnet50onCatDogDataset()
    }

    @Test
    fun resnet50v2onCatDogDatasetTest() {
        resnet50v2onCatDogDataset()
    }

    @Test
    fun resnet101onCatDogDatasetTest() {
        resnet101onCatDogDataset()
    }

    @Test
    fun resnet101v2onCatDogDatasetTest() {
        resnet101v2onCatDogDataset()
    }

    @Test
    fun resnet152onCatDogDatasetTest() {
        resnet152onCatDogDataset()
    }

    @Test
    fun resnet152v2OnCatDogDatasetTest() {
        resnet152v2OnCatDogDataset()
    }
}
