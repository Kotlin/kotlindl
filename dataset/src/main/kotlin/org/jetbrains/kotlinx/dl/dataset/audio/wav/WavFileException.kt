/*
 * Copyright 2021-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.audio.wav

/**
 * WavFile-specific exception class that represents error in reading WAV file caused by its format.
 */
public class WavFileException(message: String) : Exception(message)
