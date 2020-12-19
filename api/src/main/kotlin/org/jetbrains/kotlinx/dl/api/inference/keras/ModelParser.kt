/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras

import com.beust.klaxon.Converter
import com.beust.klaxon.JsonArray
import com.beust.klaxon.JsonValue
import org.jetbrains.kotlinx.dl.api.inference.keras.config.KerasPadding

internal class PaddingConverter : Converter {
    override fun canConvert(cls: Class<*>): Boolean {
        return cls == KerasPadding::class.java
    }

    override fun toJson(value: Any): String {
        return when (value) {
            is KerasPadding.Full -> "\"full\""
            is KerasPadding.Same -> "\"same\""
            is KerasPadding.Valid -> "\"valid\""
            is KerasPadding.ZeroPadding2D -> zeroPaddingToJsonString(value)
            else -> {
                println("[KerasPaddingConverter]: Converting unknown padding $value to JSON")
                return "$value"
            }
        }
    }

    private fun zeroPaddingToJsonString(zeroPadding2D: KerasPadding.ZeroPadding2D): String {
        with(zeroPadding2D) {
            return when (padding.size) {
                1 -> """${padding[0]}"""
                2 -> """[${padding[0]}, ${padding[1]}]"""
                4 -> """[[${padding[0]}, ${padding[1]}], [${padding[2]}, ${padding[3]}]]"""
                else -> throw IllegalArgumentException("[KerasPaddingConverter]: expected padding array with size 1, 2 or 4, got array $padding")
            }
        }
    }

    @Suppress("UNCHECKED_CAST")
    override fun fromJson(jv: JsonValue): KerasPadding? {
        // See https://github.com/tensorflow/tensorflow/blob/582c8d236cb079023657287c318ff26adb239002/tensorflow/python/keras/layers/pooling.py#L470
        // and https://github.com/tensorflow/tensorflow/blob/582c8d236cb079023657287c318ff26adb239002/tensorflow/python/keras/layers/convolutional.py#L2795
        // for detailed explaination of supported formats.
        // Could be string for `same`, `valid` or `causal`,
        // or Int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints - for ZeroPadding layer.

        val stringValue = jv.string
        if (stringValue != null) {
            return when (stringValue) {
                "same" -> KerasPadding.Same
                "valid" -> KerasPadding.Valid
                "full" -> KerasPadding.Full
                else -> throw UnsupportedOperationException("[KerasPaddingConverter]: Invalid padding $jv: the $stringValue padding is not supported!")
            }
        }

        val intValue = jv.int
        if (intValue != null) {
            return KerasPadding.ZeroPadding2D(intArrayOf(intValue))
        }

        val array = jv.array
        if (array != null) {
            assert(array.size == 2) {
                "[KerasPaddingConverter]: invalid padding shape in $jv, received ${array.size}, expected 2"
            }
            when (array[0]) {
                is Int -> {
                    assert(array[1] is Int) { "[KerasPaddingConverter]: invalid padding $jv, expected Integer at ${array[1]}" }
                    return KerasPadding.ZeroPadding2D(intArrayOf(array[0] as Int, array[1] as Int))
                }
                is JsonArray<*> -> {
                    assert(array[1] is JsonArray<*>) { "[KerasPaddingConverter]: invalid padding $jv, expected JsonArray at ${array[1]}" }
                    val firstArray = array[0] as JsonArray<Int>
                    val secondArray = array[1] as JsonArray<Int>
                    return KerasPadding.ZeroPadding2D(intArrayOf(
                        firstArray[0],
                        firstArray[1],
                        secondArray[0],
                        secondArray[1]
                    ))
                }
            }
        }

        throw UnsupportedOperationException("[KerasPaddingConverter]: Unsupported padding $jv")
    }
}