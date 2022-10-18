/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.callback

import org.jetbrains.kotlinx.dl.api.core.history.EpochTrainingEvent
import org.jetbrains.kotlinx.dl.api.core.history.TrainingHistory

/**
 * Callback for stopping training of a model when a specified amount of time has passed.
 *
 * This callback could be used to limit the training time of a model to a specified maximum duration.
 * When the specified amount of time passes, the training of the model is stopped (if it has not already been
 * finished or stopped) at the end of current epoch.
 *
 * @property [seconds] Maximum amount of time before stopping training.
 * @property [verbose] Verbosity mode.
 */
public class TimeStopping(
    // TODO: alternatively, we can use `kotlin.time.Duration` instead (after upgrading to Kotlin 1.5).
    private val seconds: Int = 86400,
    private val verbose: Boolean = false,
) : Callback() {

    private var stoppedEpoch: Int = -1
    private var stoppingTime: Long = 0

    override fun onTrainBegin() {
        stoppingTime = System.currentTimeMillis() / 1000 + seconds
    }

    override fun onTrainEnd(logs: TrainingHistory) {
        if (stoppedEpoch >= 0 && verbose) {
            this.model.logger.info {
                "Timed stopping at epoch $stoppedEpoch after training for $seconds seconds."
            }
        }
    }

    override fun onEpochEnd(epoch: Int, event: EpochTrainingEvent, logs: TrainingHistory) {
        if (System.currentTimeMillis() / 1000 >= stoppingTime) {
            this.model.stopTraining = true
            stoppedEpoch = epoch
        }
    }
}
