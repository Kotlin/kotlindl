/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.modelhub.vgg16

import org.junit.jupiter.api.Test

class VGG16TestSuite {
    @Test
    fun vgg16predictionTest() {
        vgg16prediction()
    }

    @Test
    fun vgg16noTopAdditionalTrainingTest() {
        vgg16noTopAdditionalTraining()
    }
}
