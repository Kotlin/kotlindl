/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.preprocessing

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape

/**
 * Interface for preprocessing operations.
 * @param I is a type of an input.
 * @param O is a type of an output.
 */
public interface Operation<I, O> {
    /**
     * Performs preprocessing operation on the input.
     * @param input is an input to the operation of type [I].
     * @return an output of the operation of type [O].
     */
    public fun apply(input: I): O

    /**
     * Returns shape of the output of the operation having input of shape [inputShape].
     * @param inputShape is a shape of the input.
     */
    public fun getOutputShape(inputShape: TensorShape): TensorShape
}

/**
 * Identity operation which does nothing.
 */
public class Identity<I> : Operation<I, I> {
    override fun apply(input: I): I = input
    override fun getOutputShape(inputShape: TensorShape): TensorShape = inputShape
}
