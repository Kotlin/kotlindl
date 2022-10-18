/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.integration

open class IntegrationTest {
    protected val EPOCHS = 1
    protected val TRAINING_BATCH_SIZE = 1000
    protected val TEST_BATCH_SIZE = 1000
    protected val NUM_CHANNELS = 1L
    protected val IMAGE_SIZE = 28L
    protected val SEED = 12L
    protected val EPS = 0.1
    protected val VALIDATION_BATCH_SIZE = 100
    protected val VALIDATION_RATE = 0.1
    protected val AMOUNT_OF_CLASSES = 10
}
