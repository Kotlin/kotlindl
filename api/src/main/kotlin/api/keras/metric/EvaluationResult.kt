package api.keras.metric

data class EvaluationResult(val lossValue: Double, val metrics: Map<Metrics, Double>)

