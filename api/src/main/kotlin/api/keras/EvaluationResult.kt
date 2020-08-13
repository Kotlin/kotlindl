package api.keras

import api.keras.metric.Metrics

data class EvaluationResult(val lossValue: Double, val metrics: Map<Metrics, Double>)

