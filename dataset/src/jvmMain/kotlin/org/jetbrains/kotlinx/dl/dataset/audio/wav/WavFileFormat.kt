/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.audio.wav

/**
 * Wav file format representing the specification of the WAV file that is saved in its metadata header.
 *
 * @param buffer from which the format data is read
 */
public class WavFileFormat(buffer: ByteArray) {

    public val numChannels: Int = readLittleEndian(buffer, 2, 2).toInt()

    public val sampleRate: Long = readLittleEndian(buffer, 4, 4)

    public val blockAlign: Int = readLittleEndian(buffer, 12, 2).toInt()

    private val validBits: Int = readLittleEndian(buffer, 14, 2).toInt()

    public val bytesPerSample: Int = (validBits + 7) / 8

    public val floatScale: Float

    public val floatOffset: Float

    init {
        if (bytesPerSample * numChannels != blockAlign) {
            throw WavFileException("Block Align does not agree with bytes required for validBits and number of channels")
        }
        if (numChannels == 0) {
            throw WavFileException("Number of channels specified in header is equal to zero")
        }
        if (blockAlign == 0) {
            throw WavFileException("Block Align specified in header is equal to zero")
        }
        if (validBits < 2) {
            throw WavFileException("Valid Bits specified in header is less than 2")
        } else if (validBits > 64) {
            throw WavFileException("Valid Bits specified in header is greater than 64, this is greater than a long can hold")
        }
        if (validBits > 8) {
            this.floatOffset = 0.0f
            this.floatScale = (1 shl (validBits - 1)).toFloat()
        } else {
            this.floatOffset = -1.0f
            this.floatScale = 0.5f * ((1 shl validBits) - 1)
        }
    }

    override fun toString(): String =
        "WavFileFormat(numChannels=$numChannels, sampleRate=$sampleRate, blockAlign=$blockAlign, " +
                "bytesPerSample=$bytesPerSample, floatScale=$floatScale, floatOffset=$floatOffset)"
}
