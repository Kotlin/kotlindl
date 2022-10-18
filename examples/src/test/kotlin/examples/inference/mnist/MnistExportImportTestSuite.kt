/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.inference.mnist

import org.junit.jupiter.api.Test

class MnistExportImportTestSuite {
    @Test
    fun lenetOnMnistDatasetExportImportToTxtTest() {
        lenetOnMnistDatasetExportImportToTxt()
    }

    @Test
    fun lenetOnMnistExportImportToJsonTest() {
        lenetOnMnistExportImportToJson()
    }
}
