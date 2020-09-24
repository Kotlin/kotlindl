package api.core.metric

/**
 * Represents result of evaluation on test dataset.
 *
 * @property lossValue Value of loss function on test dataset.
 * @property metrics Values of calculated metrics.
 */
data class EvaluationResult(val lossValue: Double, val metrics: Map<Metrics, Double>)

