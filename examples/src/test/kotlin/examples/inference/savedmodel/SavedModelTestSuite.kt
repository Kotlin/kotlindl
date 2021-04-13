/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.inference.savedmodel

import org.junit.jupiter.api.Test

class SavedModelTestSuite {
    @Test
    fun printOutGraphOpsTest() {
        printOutGraphOps()
    }

    @Test
    fun lenetOnMnistInferenceTest() {
        lenetOnMnistInference()
    }

    @Test
    fun lenetOnMnistInferenceWithTensorNamesTest() {
        lenetOnMnistInferenceWithTensorNames()
    }
}
