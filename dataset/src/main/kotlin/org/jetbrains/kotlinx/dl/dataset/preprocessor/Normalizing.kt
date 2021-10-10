package org.jetbrains.kotlinx.dl.dataset.preprocessor

import kotlin.math.sqrt

/**
 * This preprocessor defines normalizing operation.
 * Given mean and std for n channels, this operation normalizes each channel of the input array, i.e.
 * ```
 * output[channel] = (input[channel] - mean[channel]) / std[channel].
 * ```
 * @property [mean] an array of mean values for each channel.
 * @property [std] an array of std values for each channel.
 */
public class Normalizing : Preprocessor {
    public lateinit var mean: FloatArray
    public lateinit var std: FloatArray

    override fun apply(data: FloatArray, inputShape: ImageShape): FloatArray {
        val channels = inputShape.channels!!.toInt()
        require(mean.size == channels)
        require(std.size == inputShape.channels.toInt())

        for (i in data.indices) {
            data[i] = (data[i] - mean[i % channels]) / std[i % channels]
        }

        return data
    }
}

public fun mean(vararg arrays: FloatArray, channels: Int = 1): FloatArray {
    val result = FloatArray(3) { 0f }
    val n = arrays.sumOf { it.size / channels }

    for (floats in arrays) {
        require(floats.size % channels == 0)
        for (i in floats.indices) {
            result[i % channels] += floats[i] / n
        }
    }
    return result
}

public fun std(vararg arrays: FloatArray, channels: Int = 1): FloatArray {
    val sumSquares = FloatArray(3) { 0f }
    val sum = FloatArray(3) { 0f }
    val n = arrays.sumOf { it.size / channels }

    for (floats in arrays) {
        require(floats.size % channels == 0)
        for (i in floats.indices) {
            sumSquares[i % channels] += floats[i] * floats[i] / n
            sum[i % channels] += floats[i] / n
        }
    }
    return FloatArray(3) { sqrt(sumSquares[it] - sum[it] * sum[it]) }
}

public fun FloatArray.mean(channels: Int = 1): FloatArray = mean(this, channels = channels)
public fun FloatArray.std(channels: Int = 1): FloatArray = std(this, channels = channels)
