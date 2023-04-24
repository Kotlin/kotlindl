/*
 * Copyright 2020-2023 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.summary

import java.io.PrintStream

/**
 * Formats and prints model summary to output stream
 * By defaults prints to console
 */
public fun ModelSummary.print(out: PrintStream = System.out): Unit =
    format().forEach(out::println)

/**
 * Formats and prints model summary to output stream
 * By defaults prints to console
 */
public fun ModelWithSummary.printSummary(out: PrintStream = System.out): Unit =
    summary().print(out)
