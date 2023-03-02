/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.callback

import org.jetbrains.kotlinx.dl.api.core.history.BatchTrainingEvent
import org.jetbrains.kotlinx.dl.api.core.history.TrainingHistory

/**
 * This callback is used to stop the training if loss is not a number (NaN or INFINITY).
 */
public class TerminateOnNaN : Callback() {
    override fun onTrainBatchEnd(batch: Int, batchSize: Int, event: BatchTrainingEvent, logs: TrainingHistory) {
        val loss = event.lossValue
        if (loss.isNaN() || loss == Double.POSITIVE_INFINITY || loss == Double.NEGATIVE_INFINITY) {
            this.model.logger.info { "Batch $batch: Invalid loss $loss, terminating training" }
            this.model.stopTraining = true
        }
    }
}
