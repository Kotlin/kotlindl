/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.preprocessing

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import kotlin.math.sqrt

/**
 * This preprocessor defines normalizing operation.
 * Given mean and std for n channels, this operation normalizes each channel of the input array, i.e.
 * ```
 * output[channel] = (input[channel] - mean[channel]) / std[channel].
 * ```
 * @property [mean] an array of mean values for each channel.
 * @property [std] an array of std values for each channel.
 * @property [channelsLast] true is channel dimension is the last one, false if it's the first one.
 */
public class Normalizing : FloatArrayOperation() {
    public lateinit var mean: FloatArray
    public lateinit var std: FloatArray
    public var channelsLast: Boolean = true

    override fun applyImpl(data: FloatArray, shape: TensorShape): FloatArray {
        val channels = if (channelsLast) shape.tail().last().toInt() else shape.head().toInt()
        require(mean.size == channels) {
            "Expected to get one mean value for each image channel. " +
                    "However ${mean.size} values was given for image with $channels channels."
        }
        require(std.size == channels) {
            "Expected to get one std value for each image channel. " +
                    "However ${std.size} values was given for image with $channels channels."
        }

        for (i in data.indices) {
            val c = if (channelsLast) i % channels else i / (shape.numElements() / channels).toInt()
            data[i] = (data[i] - mean[c]) / std[c]
        }

        return data
    }
}

/**
 * Computes mean value for each channel of the provided arrays.
 *
 * NOTE: might be migrated to multik in the future.
 *
 * @param [arrays] input arrays. Size of each array should be divisible by the passed [channels] number.
 * @param [channels] number of channels to compute mean value for.
 * @return an array of size [channels] containing mean value for each channel.
 */
public fun mean(vararg arrays: FloatArray, channels: Int = 1): FloatArray {
    val result = FloatArray(3) { 0f }
    val n = arrays.sumOf { it.size / channels }

    for (floats in arrays) {
        require(floats.size % channels == 0) {
            "Expected input array size to be divisible by the number of channels. " +
                    "However got an array of size ${floats.size} while the number of channels is $channels"
        }
        for (i in floats.indices) {
            result[i % channels] += floats[i] / n
        }
    }
    return result
}

/**
 * Computes std value for each channel of the provided arrays.
 *
 * NOTE: might be migrated to multik in the future.
 *
 * @param [arrays] input arrays. Size of each array should be divisible by the passed [channels] number.
 * @param [channels] number of channels to compute std value for.
 * @return an array of size [channels] containing std value for each channel.
 */
public fun std(vararg arrays: FloatArray, channels: Int = 1): FloatArray {
    val sumSquares = FloatArray(3) { 0f }
    val sum = FloatArray(3) { 0f }
    val n = arrays.sumOf { it.size / channels }

    for (floats in arrays) {
        require(floats.size % channels == 0) {
            "Expected input array size to be divisible by the number of channels. " +
                    "However got an array of size ${floats.size} while the number of channels is $channels"
        }
        for (i in floats.indices) {
            sumSquares[i % channels] += floats[i] * floats[i] / n
            sum[i % channels] += floats[i] / n
        }
    }
    return FloatArray(3) { sqrt(sumSquares[it] - sum[it] * sum[it]) }
}

/**
 * Computes mean value for each channel of the array. Array size should be divisible by the passed [channels] number.
 *
 * NOTE: might be migrated to multik in the future.
 *
 * @param [channels] number of channels to compute mean value for.
 * @return an array of size [channels] containing mean value for each channel.
 */
public fun FloatArray.mean(channels: Int = 1): FloatArray = mean(this, channels = channels)

/**
 * Computes std value for each channel of the array. Array size should be divisible by the passed [channels] number.
 *
 * NOTE: might be migrated to multik in the future.
 *
 * @param [channels] number of channels to compute std value for.
 * @return an array of size [channels] containing std value for each channel.
 */
public fun FloatArray.std(channels: Int = 1): FloatArray = std(this, channels = channels)
