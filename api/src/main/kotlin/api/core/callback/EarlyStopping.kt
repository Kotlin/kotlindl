package api.core.callback

import api.core.history.EpochTrainingEvent
import api.core.history.TrainingHistory
import java.util.function.BiFunction
import java.util.logging.Level
import java.util.logging.Logger
import kotlin.math.abs

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
 */
public class EarlyStopping : Callback() {
    private var wait = 0
    private var stoppedEpoch = 0

    /**
     * Quantity to be monitored. Default is "val_loss".
     */
    private var monitor: String = "val_loss"

    /**
     * Minimum change in the monitored quantity to qualify as an improvement,
     * i.e. an absolute change of less than min_delta, will count as no
     * improvement. Default is 0.
     */
    private var minDelta = 0.0

    /**
     * Number of epochs with no improvement after which training will be
     * stopped. Default is 0.
     */
    private var patience = 0

    /**
     * verbosity mode. Default is false.
     */
    private var verbose = false

    /**
     * One of {"auto", "min", "max"}. In min mode, training will stop when the
     * quantity monitored has stopped decreasing; in max mode it will stop when
     * the quantity monitored has stopped increasing; in auto mode, the
     * direction is automatically inferred from the name of the monitored
     * quantity. Default is Mode.auto.
     */
    private var mode: EarlyStoppingMode = EarlyStoppingMode.AUTO

    /**
     * Baseline value for the monitored quantity. Training will stop if the
     * model doesn't show improvement over the baseline.
     */
    private var baseline: Double = 0.001

    /**
     * Flag indicating whether to restore model weights from the epoch with the
     * best value of the monitored quantity. If false (default), the model
     * weights obtained at the last step of training are used.
     */
    private var restoreBestWeights = false
    private var best = 0.0
    private var monitorGreater = false
    private var monitorOp: BiFunction<Number, Number, Boolean>? = null

    /**
     * Create an EarlyStopping Callback
     *
     * @param monitor Quantity to be monitored.
     * @param minDelta Minimum change in the monitored quantity to qualify as an
     * improvement, i.e. an absolute change of less than min_delta, will count
     * as no improvement.
     * @param patience Number of epochs with no improvement after which training
     * will be stopped.
     * @param verbose verbosity mode.
     * @param mode In min mode, training will stop when the quantity monitored
     * has stopped decreasing; in max mode it will stop when the quantity
     * monitored has stopped increasing; in auto mode, the direction is
     * automatically inferred from the name of the monitored quantity.
     * @param baseline Baseline value for the monitored quantity. Training will
     * stop if the model doesn't show improvement over the baseline.
     * @param restoreBestWeights Whether to restore model weights from the epoch
     * with the best value of the monitored quantity. If false, the model
     * weights obtained at the last step of training are used.
     */
    fun setUp(
        monitor: String,
        minDelta: Double, patience: Int, verbose: Boolean, mode: EarlyStoppingMode,
        baseline: Double, restoreBestWeights: Boolean
    ): EarlyStopping {
        this.monitor = monitor
        this.minDelta = abs(minDelta)
        this.patience = patience
        this.verbose = verbose
        this.mode = mode
        this.baseline = baseline
        this.restoreBestWeights = restoreBestWeights

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
            else -> if (this.monitor == "acc") {
                monitorOp = BiFunction { a: Number, b: Number -> a.toDouble() > b.toDouble() }
                monitorGreater = true
                best = Double.MAX_VALUE
            } else {
                monitorOp = BiFunction { a: Number, b: Number -> a.toDouble() < b.toDouble() }
                this.minDelta *= -1.0
                best = Double.MIN_VALUE
            }
        }

        return this
    }

    override fun onTrainBegin() {
        wait = 0
        stoppedEpoch = 0
        best =
            if (baseline != null) baseline!! else if (monitorGreater) Double.POSITIVE_INFINITY else Double.NEGATIVE_INFINITY
    }

    override fun onEpochEnd(epoch: Int, event: EpochTrainingEvent, logs: TrainingHistory) {
        val current: Number = getMonitorValue(event, monitor) ?: return
        if (monitorOp!!.apply(current.toDouble() - minDelta, best)) {
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
                        Logger.getLogger(EarlyStopping::class.java.name).log(
                            Level.INFO,
                            "Restoring model weights from the end of the best epoch."
                        )
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

    private fun getMonitorValue(logs: EpochTrainingEvent, monitor: String): Number? {
        val monitorValue = logs.lossValue // TODO: extract specific monitor metric instead default
        if (monitorValue == null) {
            this.model.logger.warn {
                "Early stopping conditioned on metric $monitor which is not available. Available metrics are: $logs"
            }
        }
        return monitorValue
    }
}