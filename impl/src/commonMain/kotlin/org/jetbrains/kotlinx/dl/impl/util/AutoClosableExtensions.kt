/*
 * Copyright 2022-2023 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.util

/**
 * Executes the given [block] function on this resource list and closes the resources correctly
 * even when exception is thrown from the block. Similar to [kotlin.use] extension.
 */
public fun <A : AutoCloseable, R> List<A>.use(block: (List<A>) -> R): R {
    if (isEmpty()) return block(this)

    var exception: Throwable? = null
    try {
        return block(this)
    } catch (e: Throwable) {
        exception = e
        throw e
    } finally {
        closeSafely(exception)
    }
}

/**
 * Executes the given [block] function on this resources map and closes the resources correctly
 * even when exception is thrown from the block. Similar to [kotlin.use] extension.
 */
public fun <K, A : AutoCloseable, R> Map<K, A>.use(block: (Map<K, A>) -> R): R {
    if (isEmpty()) return block(this)

    var exception: Throwable? = null
    try {
        return block(this)
    } catch (e: Throwable) {
        exception = e
        throw e
    } finally {
        values.closeSafely(exception)
    }
}

private fun <A : AutoCloseable> Collection<A>.closeSafely(cause: Throwable?) {
    forEach {
        try {
            it.close()
        } catch (e: Throwable) {
            cause?.addSuppressed(e)
        }
    }
}