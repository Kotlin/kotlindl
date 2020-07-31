package api.keras

import api.getDType
import api.keras.shape.TensorShape
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.ReduceSum
import org.tensorflow.op.core.Squeeze
import org.tensorflow.op.math.Mean
import util.Triada

const val EPSILON = 1e-7f

fun <T : Number> Kmean(tf: Ops, x: Operand<T>): Operand<T> {
    return Kmean(tf, x, null, false)
}

fun <T : Number> Kmean(
    tf: Ops,
    x: Operand<T>,
    axis: Operand<Int>
): Operand<T> {
    return Kmean(tf, x, axis, false)
}

fun <T : Number> Kmean(tf: Ops, x: Operand<T>, keepDims: Boolean): Operand<T> {
    return Kmean(tf, x, null, keepDims)
}

fun <T : Number> Kmean(
    tf: Ops,
    x: Operand<T>,
    axis: Operand<Int>?,
    keepDims: Boolean
): Operand<T> {
    var localAxis = axis

    if (localAxis == null) {
        localAxis = KallAxis(tf, x)
    }
    return tf.math.mean(x, localAxis, Mean.keepDims(keepDims))
}


fun KallAxis(tf: Ops, op: Operand<*>): Operand<Int> {
    val ranks: IntArray = KallAxis(op)
    return tf.constant(ranks)

}

fun KallAxis(op: Operand<*>): IntArray {
    var rank = op.asOutput().shape().numDimensions()
    if (rank == 0) return intArrayOf(1)
    val ranks = IntArray(rank)
    for (i in 0 until rank) {
        ranks[i] = i
    }
    return ranks
}

fun <T : Number> Kone(tf: Ops): Operand<T> {
    return tf.dtypes.cast(tf.constant(1), getDType<T>())
}

fun <T : Number> epsilon(tf: Ops): Operand<T> {
    return tf.constant(EPSILON) as Operand<T>
}

fun <T : Number> computeWeightedMetric(
    tf: Ops,
    losses: Operand<T>,
    reduction: Reduction = Reduction.AUTO
): Operand<T> {
    val initSampleWeight = Kone<T>(tf)
    val result = squeezeOrExpandDimensions<T>(tf, null, losses, initSampleWeight)
    var _losses = result.getLosses()
    val changedSampleWeight = result.getSampleWeights()

    val weighted_losses: Operand<T> = tf.math.mul(_losses, changedSampleWeight)
    _losses = reduceWeightedLoss(tf, weighted_losses, reduction)
    return _losses
}

fun <T : Number> computeWeightedLoss(
    tf: Ops,
    losses: Operand<T>,
    reduction: Reduction = Reduction.AUTO
): Operand<T> {
    val initSampleWeight = Kone<T>(tf)
    val result = squeezeOrExpandDimensions<T>(tf, null, losses, initSampleWeight)
    var _losses = result.getLosses()
    val changedSampleWeight = result.getSampleWeights()

    val weighted_losses: Operand<T> = tf.math.mul(_losses, changedSampleWeight)
    _losses = reduceWeightedLoss(tf, weighted_losses, reduction)
    return _losses
}

fun <T : Number> squeezeOrExpandDimensions(
    tf: Ops,
    yTrue: Operand<T>?,
    yPred: Operand<T>,
    sampleWeight: Operand<T>?
): Triada<T> {
    var sampleWeight = sampleWeight
    var tuple = Triada(yTrue, yPred)
    val ypredShape = yPred.asOutput().shape()
    val ypredRank = ypredShape.numDimensions()
    if (yTrue != null) {
        val ytrueShape = yTrue.asOutput().shape()
        val ytrueRank = ytrueShape.numDimensions()
        if (ytrueRank != -1 && ypredRank != -1) {
            // Use static rank for `y_true` and `y_pred`.
            if (ypredRank - ytrueRank != 1 || ypredShape.size(-1).toInt() == 1) {
                //y_true, y_pred = confusion_matrix.remove_squeezable_dimensions(y_true, y_pred)
                tuple = removeSqueezableDimensions(tf, yTrue, yPred)
            }
        } else { // use dynamic rank
            tuple = removeSqueezableDimensions(tf, yTrue, yPred)
        }
    }
    if (sampleWeight == null) {
        return tuple
    }
    val weightsShape = sampleWeight.asOutput().shape()
    val weightsRank = weightsShape.numDimensions()
    if (weightsRank == 0) { // scalar
        return Triada(yTrue, yPred, sampleWeight)
    }
    if (ypredRank != -1 && weightsRank != -1) {
        if (weightsRank - ypredRank == 1) {
            sampleWeight = tf.squeeze<T>(sampleWeight)
        } else if (ypredRank - weightsRank == 1) {
            sampleWeight = tf.expandDims(sampleWeight, tf.constant(-1))
        }
        return Triada(yTrue, yPred, sampleWeight)
    }
    // Use dynamic rank.
    val weightsRankTensor = tf.rank(sampleWeight)
    val rankDiff = tf.math.sub(weightsRankTensor, tf.rank(yPred))
    sampleWeight = tf.selectV2(
        tf.math.equal(weightsRankTensor, tf.constant(0)),
        sampleWeight,
        maybeAdjustWeights(tf, sampleWeight, rankDiff)
    )
    return Triada(yTrue, yPred, sampleWeight)

}

fun <T : Number> maybeAdjustWeights(tf: Ops, sampleWeight: Operand<T>, rankDiff: Operand<Int>): Operand<T> {
    return tf.selectV2(
        tf.math.equal(rankDiff, tf.constant(1)),
        tf.squeeze(sampleWeight, Squeeze.axis(listOf(-1L))),
        maybeExpandWeights(tf, sampleWeight, rankDiff)
    )
}

fun <T> maybeExpandWeights(tf: Ops, sampleWeight: Operand<T>, rankDiff: Operand<Int>): Operand<T> {
    return tf.selectV2(
        tf.math.equal(rankDiff, tf.constant(-1)),
        tf.expandDims(sampleWeight, tf.constant(-1)), sampleWeight
    )
}

fun <T : Number> removeSqueezableDimensions(tf: Ops, labels: Operand<T>, predictions: Operand<T>): Triada<T> {
    var _labels = labels
    var _predictions = predictions

    tf.withSubScope("removeSqueezableDimensions")
    val predictionsShape: Shape = predictions.asOutput().shape()
    val predictionsRank = predictionsShape.numDimensions()
    val labelsShape: Shape = labels.asOutput().shape()
    val labelsRank = labelsShape.numDimensions()

    if (predictionsRank != -1 && labelsRank != -1) {
        // Use static rank.
        val rankDiff = predictionsRank - labelsRank
        val expectedRankDiff = 0
        if (rankDiff == expectedRankDiff + 1 && isCompatible(predictionsShape.size(-1), 1)) {
            _predictions = tf.squeeze(_predictions)
        } else if (rankDiff == expectedRankDiff - 1 && isCompatible(labelsShape.size(-1), 1)) {
            _labels = tf.squeeze(_labels)
        }
        return Triada(_labels, _predictions)
    }
    // Use dynamic rank.
    // Use dynamic rank.
    val rankDiff: Operand<*> =
        tf.math.sub(tf.rank(predictions), tf.rank(labels))
    if (predictionsRank == -1 && isCompatible(predictionsShape.size(-1), 1)) {
        /**
         * TODO, if we ever get a select that does lazy evaluation, but for
         * now do the tf.squeeze predictions = tf.select(
         * tf.math.equal(tf.constant(expectedRankDiff+1),rankDiff ),
         * tf.squeeze(predictions, Squeeze.axis(Arrays.asList(-1L))),
         * predictions ); *
         */
        _predictions = tf.squeeze(_predictions, Squeeze.axis(listOf(-1L)))
    }
    if (labelsRank == -1 && isCompatible(labelsShape.size(-1), 1)) {
        /**
         * TODO, if we ever get a select that does lazy evaluation labels =
         * tf.select( tf.math.equal(tf.constant(expectedRankDiff+1),rankDiff
         * ), tf.squeeze(labels, Squeeze.axis(Arrays.asList(-1L))),
         * predictions ); *
         */
        _labels = tf.squeeze(_labels, Squeeze.axis(listOf(-1L)))
    }
    return Triada(_labels, _predictions)
}

fun isCompatible(dim: Long, otherDim: Long): Boolean {
    return dim == -1L || otherDim == -1L || dim == otherDim
}

fun <T : Number> reduceWeightedLoss(tf: Ops, weightedLosses: Operand<T>, reduction: Reduction): Operand<T> {
    var loss: Operand<T>

    if (reduction === Reduction.NONE) {
        loss = weightedLosses
    } else {
        loss =
            tf.reduceSum(weightedLosses, KallAxis(tf, weightedLosses), ReduceSum.keepDims(java.lang.Boolean.FALSE))
        if (reduction === Reduction.AUTO || reduction === Reduction.SUM_OVER_BATCH_SIZE) {
            loss = safeMean(
                tf,
                loss,
                TensorShape(weightedLosses.asOutput().shape()).numElements()
            )
        }
    }
    return loss
}

fun <T : Number> safeMean(tf: Ops, loss: ReduceSum<T>, numElements: Long): Operand<T> {
    if (loss.asOutput().shape().numDimensions() == 0) {
        return tf.math.divNoNan(loss, tf.constant(numElements.toFloat()) as Operand<T>)
    } else {
        val totalLoss = tf.reduceSum(loss, KallAxis(tf, loss))
        return tf.math.divNoNan(totalLoss, tf.constant(numElements.toFloat()) as Operand<T>)
    }

}

enum class Reduction {
    AUTO, NONE, SUM, SUM_OVER_BATCH_SIZE
}
