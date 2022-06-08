/*
 * Copyright 2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.util

/**
 * Create a new read-only map from a list of pairs, if both values in the pair are not null.
 *
 * @param [mapping] pairs of keys and values to put into the map
 * @return map with the provided keys and values, excluding nulls
 * @see kotlin.collections.mapOf(Pair[])
 */
public fun <K, V> mapOfNotNull(vararg mapping: Pair<K?, V?>): Map<K, V> {
    return buildMap {
        mapping.forEach { (k, v) ->
            if (k != null && v != null) {
                put(k, v)
            }
        }
    }
}