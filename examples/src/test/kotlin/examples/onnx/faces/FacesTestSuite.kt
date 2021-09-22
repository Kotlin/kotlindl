/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.faces

import examples.onnx.objectdetection.ssd.objectDetectionSSD
import examples.onnx.objectdetection.ssd.predictionSSD
import org.junit.jupiter.api.Test

class FacesTestSuite {
    @Test
    fun predictionSSDTest() {
        predictionSSD()
    }

    @Test
    fun objectDetectionSSDTest() {
        objectDetectionSSD()
    }
}
