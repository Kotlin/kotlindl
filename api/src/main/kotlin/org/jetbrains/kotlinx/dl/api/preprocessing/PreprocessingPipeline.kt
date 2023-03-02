/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.preprocessing

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape


/**
 * This class is a special type of Operation which is used to build typesafe pipeline of preprocessing operations.
 * Kudos to [@juliabeliaeva](https://github.com/juliabeliaeva) for the idea.
 */
public class PreprocessingPipeline<I, M, O>(
    private val firstOp: Operation<I, M>,
    private val secondOp: Operation<M, O>
) : Operation<I, O> {
    override fun apply(input: I): O = secondOp.apply(firstOp.apply(input))
    override fun getOutputShape(inputShape: TensorShape): TensorShape {
        return secondOp.getOutputShape(firstOp.getOutputShape(inputShape))
    }
}

/**
 * An entry point for building preprocessing pipeline.
 */
public fun <I> pipeline(): Identity<I> = Identity()
