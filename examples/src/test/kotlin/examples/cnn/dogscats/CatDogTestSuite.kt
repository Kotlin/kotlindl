/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.cnn.dogscats

import org.junit.jupiter.api.Test

class CatDogTestSuite {
    @Test
    fun resnet50onCatDogDatasetTest() {
        resnet50onDogsVsCatsDataset()
    }

    @Test
    fun resnet50v2onCatDogDatasetTest() {
        resnet50v2onDogsVsCatsDataset()
    }

    @Test
    fun resnet101onCatDogDatasetTest() {
        resnet101onDogsVsCatsDataset()
    }

    @Test
    fun resnet101v2onCatDogDatasetTest() {
        resnet101v2onDogsVsCatsDataset()
    }

    @Test
    fun resnet152onCatDogDatasetTest() {
        resnet152onDogsVsCatsDataset()
    }

    @Test
    fun resnet152v2OnCatDogDatasetTest() {
        resnet152v2onDogsVsCatsDataset()
    }
}
