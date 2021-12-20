/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.callback

import org.jetbrains.kotlinx.dl.api.core.history.EpochTrainingEvent
import org.jetbrains.kotlinx.dl.api.core.history.TrainingHistory
import java.util.function.BiFunction
import kotlin.reflect.KProperty1

/**
 * This enum describes a few strategies of training stopping.
 */
public enum class EarlyStoppingMode {
    /**
     * In this mode the direction is
     * automatically inferred from the name of the monitored quantity.
     */
    AUTO,

    /**
     * In this mode, training will stop when the quantity monitored
     * has stopped decreasing.
     */
    MIN,

    /**
     * In this mode, training will stop when the quantity
     * monitored has stopped increasing.
     */
    MAX
}

/**
 * This callback stops training when a monitored metric has stopped improving.
 *
 * Assuming the goal of a training is to minimize the loss. With this, the
 * metric to be monitored would be `'loss'`, and mode would be `'min'`. A
 * `model.fit()` training loop will check at end of every epoch whether
 * the loss is no longer decreasing, considering the `min_delta` and
 * `patience` if applicable. Once it's found no longer decreasing,
 * `model.stop_training` is marked True and the training terminates.
 *
 * The quantity to be monitored needs to be available in `logs`.
 * To make it so, pass the loss or metrics at `model.compile()`.
 *
 * @property [monitor] Quantity to be monitored.
 * @property [minDelta] Minimum change in the monitored quantity to qualify as an
 * improvement, i.e. an absolute change of less than min_delta, will count
 * as no improvement.
 * @property [patience] Number of epochs with no improvement after which training
 * will be stopped.
 * @property [verbose] Verbosity mode.
 * @property [mode] One of {"auto", "min", "max"}. In min mode, training will stop when the quantity monitored
 * has stopped decreasing; in max mode it will stop when the quantity
 * monitored has stopped increasing; in auto mode, the direction is
 * automatically inferred from the name of the monitored quantity.
 * @property [baseline] Baseline value for the monitored quantity. Training will
 * stop if the model doesn't show improvement over the baseline.
 * @property [restoreBestWeights] Whether to restore model weights from the epoch
 * with the best value of the monitored quantity. If false, the model
 * weights obtained at the last step of training are used.
 *
 * @constructor Creates an EarlyStopping Callback.
 */
public class EarlyStopping(
    private var monitor: KProperty1<EpochTrainingEvent, Double?> = EpochTrainingEvent::lossValue,
    private var minDelta: Double = 0.0,
    private var patience: Int = 0,
    private var verbose: Boolean = false,
    private var mode: EarlyStoppingMode = EarlyStoppingMode.AUTO,
    private var baseline: Double = 0.001,
    private val restoreBestWeights: Boolean = false
) : Callback() {
    private var wait = 0

    private var stoppedEpoch = 0

    private var best = 0.0

    private var monitorGreater = false

    private var monitorOp: BiFunction<Number, Number, Boolean>? = null

    init {
        require(minDelta >= 0.0 && baseline >= 0.0)

        when (mode) {
            EarlyStoppingMode.MIN -> {
                monitorOp = BiFunction { a: Number, b: Number -> a.toDouble() < b.toDouble() }
                this.minDelta *= -1.0
                best = Double.MAX_VALUE
            }
            EarlyStoppingMode.MAX -> {
                monitorOp = BiFunction { a: Number, b: Number -> a.toDouble() > b.toDouble() }
                monitorGreater = true
                best = Double.MIN_VALUE
            }
            // If metric
            else -> if (this.monitor == EpochTrainingEvent::metricValues || this.monitor == EpochTrainingEvent::valMetricValues) { // TODO: correctly handle the case with monitoring of multiple metrics
                monitorOp = BiFunction { a: Number, b: Number -> a.toDouble() > b.toDouble() }
                monitorGreater = true
                best = Double.MAX_VALUE
            }
            // If loss
            else {
                monitorOp = BiFunction { a: Number, b: Number -> a.toDouble() < b.toDouble() }
                this.minDelta *= -1.0
                best = Double.MIN_VALUE
            }
        }
    }

    override fun onTrainBegin() {
        wait = 0
        stoppedEpoch = 0
        best =
            if (baseline != null) baseline else if (monitorGreater) Double.POSITIVE_INFINITY else Double.NEGATIVE_INFINITY
    }

    override fun onEpochEnd(epoch: Int, event: EpochTrainingEvent, logs: TrainingHistory) {
        val current: Number = getMonitorValue(event, monitor) ?: return
        if ((monitorOp ?: return).apply(current.toDouble() - minDelta, best)) {
            best = current.toDouble()
            wait = 0
            if (restoreBestWeights) {
                // TODO this.bestWeights = this.model.getWeights();
            }
        } else {
            wait++
            if (wait > patience) {
                stoppedEpoch = epoch
                this.model.stopTraining = true
                if (restoreBestWeights) {
                    if (verbose) {
                        this.model.logger.info { "Restoring model weights from the end of the best epoch." }
                    }
                    // TODO this.model.setWeights(this.bestWeights)
                }
            }
        }
    }

    override fun onTrainEnd(logs: TrainingHistory) {
        if (stoppedEpoch > 0 && verbose) {
            this.model.logger.info {
                "Epoch ${stoppedEpoch + 1}: early stopping event! "
            }
        }
    }

    private fun getMonitorValue(logs: EpochTrainingEvent, monitor: KProperty1<EpochTrainingEvent, Double?>): Number? {
        val monitorValue = monitor.get(logs)
        if (monitorValue == null) {
            this.model.logger.warn {
                "Early stopping conditioned on metric $monitor which is not available. Available metrics are: $logs"
            }
        }
        return monitorValue
    }
}
