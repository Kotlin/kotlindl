/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras

import com.beust.klaxon.JsonArray
import com.beust.klaxon.JsonObject
import com.beust.klaxon.Klaxon
import com.beust.klaxon.Parser
import org.jetbrains.kotlinx.dl.api.inference.keras.config.KerasPadding
import org.jetbrains.kotlinx.dl.api.inference.keras.config.LayerConfig
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

internal class PaddingConverterTest {
    private val klaxon = Klaxon().converter(PaddingConverter())
    private val parser = Parser.default()

    @Test
    fun parseStringValues() {
        val paddingValidString = """{"padding": "valid"}"""
        var layerConfig: LayerConfig? = klaxon.parse<LayerConfig>(paddingValidString)
        assertTrue { layerConfig!!.padding is KerasPadding.Valid }

        val paddingSameString = """{"padding": "same"}"""
        layerConfig = klaxon.parse<LayerConfig>(paddingSameString)
        assert(layerConfig!!.padding is KerasPadding.Same)

        val paddingFullString = """{"padding": "full"}"""
        layerConfig = klaxon.parse<LayerConfig>(paddingFullString)
        assert(layerConfig!!.padding is KerasPadding.Full)
    }

    @Test
    fun parseZeroPaddingValues() {
        val oneValueString = """{"padding": 1}"""
        (klaxon.parse<LayerConfig>(oneValueString)!!.padding as KerasPadding.ZeroPadding2D).apply {
            assertArrayEquals(intArrayOf(1), padding)
        }

        val twoValuesString = """{"padding": [1, 2]}"""
        (klaxon.parse<LayerConfig>(twoValuesString)!!.padding as KerasPadding.ZeroPadding2D).apply {
            assertArrayEquals(intArrayOf(1, 2), padding)
        }

        val twoArraysString = """{"padding": [[1, 2], [3, 4]]}"""
        (klaxon.parse<LayerConfig>(twoArraysString)!!.padding as KerasPadding.ZeroPadding2D).apply {
            assertArrayEquals(intArrayOf(1, 2, 3, 4), padding)
        }
    }

    @Test
    fun serializeStringPaddings() {
        var serializedConfig: String

        val paddingValidConfig = LayerConfig(padding = KerasPadding.Valid)
        serializedConfig = klaxon.toJsonString(paddingValidConfig)
        var jsonObject: JsonObject = parser.parse(StringBuilder(serializedConfig)) as JsonObject
        assertEquals("valid", jsonObject["padding"])

        val paddingSameConfig = LayerConfig(padding = KerasPadding.Same)
        serializedConfig = klaxon.toJsonString(paddingSameConfig)
        jsonObject = parser.parse(StringBuilder(serializedConfig)) as JsonObject
        assertEquals("same", jsonObject["padding"])

        val paddingFullConfig = LayerConfig(padding = KerasPadding.Full)
        serializedConfig = klaxon.toJsonString(paddingFullConfig)
        jsonObject = parser.parse(StringBuilder(serializedConfig)) as JsonObject
        assertEquals("full", jsonObject["padding"])
    }

    @Test
    fun serializeZeroPaddingValues() {
        var serializedConfig: String

        var layerConfig = LayerConfig(padding = KerasPadding.ZeroPadding2D(1))
        serializedConfig = klaxon.toJsonString(layerConfig)
        var jsonObject: JsonObject = parser.parse(StringBuilder(serializedConfig)) as JsonObject
        assertEquals(1, jsonObject["padding"])

        layerConfig = LayerConfig(padding = KerasPadding.ZeroPadding2D(intArrayOf(1, 2)))
        serializedConfig = klaxon.toJsonString(layerConfig)
        jsonObject = parser.parse(StringBuilder(serializedConfig)) as JsonObject
        assertArrayEquals(intArrayOf(1, 2), jsonObject.array<Int>("padding")!!.toIntArray())

        layerConfig = LayerConfig(
            padding = KerasPadding.ZeroPadding2D(
                arrayOf(
                    intArrayOf(1, 2),
                    intArrayOf(3, 4)
                )
            )
        )
        serializedConfig = klaxon.toJsonString(layerConfig)
        jsonObject = parser.parse(StringBuilder(serializedConfig)) as JsonObject
        val parsedArray = jsonObject.array<JsonArray<Int>>("padding")!!
        assertArrayEquals(intArrayOf(1, 2), parsedArray[0].toIntArray())
        assertArrayEquals(intArrayOf(3, 4), parsedArray[1].toIntArray())
    }
}
