package org.jetbrains.kotlinx.dl.example.app

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.impl.preprocessing.FloatArrayOperation
import org.jetbrains.kotlinx.dl.impl.util.argmax
import kotlin.math.exp

class Softmax : FloatArrayOperation() {
    override fun applyImpl(data: FloatArray, shape: TensorShape): FloatArray {
        val logits = data.copyOf()
        val max = logits[logits.argmax()]
        var sum = 0.0f

        for (i in logits.indices) {
            logits[i] = exp(logits[i] - max)
            sum += logits[i]
        }

        if (sum != 0.0f) {
            for (i in logits.indices) {
                logits[i] /= sum
            }
        }

        return logits
    }
}
