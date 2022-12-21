/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.audio.wav

import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStream

private const val HEADER_LENGTH = 12

private const val CHUNK_HEADER_LENGTH = 8

private const val FMT_CHUNK_ID: Long = 0x20746D66

private const val DATA_CHUNK_ID: Long = 0x61746164

private const val RIFF_CHUNK_ID: Long = 0x46464952

private const val RIFF_TYPE_ID: Long = 0x45564157

/**
 * Class for reading WAV audio files. The file opened as WAV file can be read
 * only once and the following reading procedures will result in reading empty buffer.
 *
 * Based on code written by [Andrew Greensted](http://www.labbookpages.co.uk/)
 * but modified to more Kotlin idiomatic way with only read option for simplicity.
 *
 * @property bufferSize is a size of a buffer to read from given file when reading next frames.
 * @constructor creates [WavFile]
 *
 * @param file to read the WAV file data from
 */
public class WavFile(
    file: File,
    private val bufferSize: Int = 4096
) : AutoCloseable {

    private enum class IOState {
        READING,
        CLOSED
    }

    /** Remaining frames that can be read to some external buffer from WAV file */
    public val remainingFrames: Long get() = frames - frameCounter

    /** File format specification with values from [WavFileFormat] */
    public val format: WavFileFormat

    /** Number of frames present in full WAV file */
    public val frames: Long

    private var ioState = IOState.READING

    private val inputStream = FileInputStream(file)

    private val buffer = ByteArray(bufferSize)

    private var bufferPointer = 0

    private var bytesRead = 0

    private var frameCounter = 0L

    init {
        readWavHeader(file, inputStream, buffer)

        var chunkSize: Long
        var fileFormatChunk: WavFileFormat? = null
        var numFrames: Long?

        while (true) {
            bytesRead = inputStream.read(buffer, 0, CHUNK_HEADER_LENGTH)
            if (bytesRead != CHUNK_HEADER_LENGTH) {
                throw WavFileException("Could not read chunk header")
            }

            val chunkID = readLittleEndian(buffer, 0, 4)
            chunkSize = readLittleEndian(buffer, 4, 4)

            var numChunkBytes = if (chunkSize % 2 == 1L) chunkSize + 1 else chunkSize
            if (chunkID == FMT_CHUNK_ID) {
                bytesRead = inputStream.read(buffer, 0, 16)

                val compressionCode = readLittleEndian(buffer, 0, 2).toInt()
                if (compressionCode != 1) {
                    throw WavFileException("Compression Code $compressionCode not supported")
                }

                fileFormatChunk = WavFileFormat(buffer)

                numChunkBytes -= 16
                if (numChunkBytes > 0) {
                    inputStream.skip(numChunkBytes)
                }
            } else if (chunkID == DATA_CHUNK_ID) {
                val format = fileFormatChunk ?: throw WavFileException("Data chunk found before Format chunk")
                if (chunkSize % format.blockAlign != 0L) {
                    throw WavFileException("Data chunk size is not multiple of Block Align")
                }
                numFrames = chunkSize / format.blockAlign
                break
            } else {
                inputStream.skip(numChunkBytes)
            }
        }

        this.format = fileFormatChunk ?: throw WavFileException("Did not find a Format chunk")
        this.frames = numFrames ?: throw WavFileException("Did not find a Data chunk")
    }

    /** */
    public override fun close() {
        ioState = IOState.CLOSED
        inputStream.close()
    }

    /**
     * Read all remaining frames from WAV file and return them as an array of
     * results for each of the channels of input file.
     *
     * @return Array with sound data for each channel
     */
    public fun readRemainingFrames(): Array<FloatArray> {
        val count = remainingFrames
        if (count > Int.MAX_VALUE) {
            throw WavFileException("Cannot read more at once than array of size ${Int.MAX_VALUE}")
        }
        val buffer = Array(format.numChannels) { FloatArray(count.toInt()) }
        val readCount = readFrames(buffer, count.toInt())
        check(readCount == count.toInt()) {
            "Internal error: Should read all remaining data from wav file."
        }
        return buffer
    }

    /**
     * Read some number of frames from a specific offset in the buffer into a multidimensional
     * float array.
     *
     * @param returnBuffer the buffer to read samples into
     * @param count the number of frames to read
     * @param offset the buffer offset to read from
     * @return the number of frames read
     */
    public fun readFrames(returnBuffer: Array<FloatArray>, count: Int, offset: Int = 0): Int {
        var myOffset = offset
        if (ioState != IOState.READING) {
            throw IOException("Cannot read from closed WavFile instance")
        }
        for (f in 0 until count) {
            if (frameCounter == frames) {
                return f
            }
            for (c in 0 until format.numChannels) {
                returnBuffer[c][myOffset] = format.floatOffset + readSingleSample().toFloat() / format.floatScale
            }
            myOffset++
            frameCounter++
        }
        return count
    }

    /**
     * Read a single sample from the buffer.
     *
     * @return the sample read
     * @throws IOException Signals that an I/O exception has occurred
     * @throws WavFileException a WavFile-specific exception
     */
    private fun readSingleSample(): Long {
        var resultSample = 0L
        for (b in 0 until format.bytesPerSample) {
            if (bufferPointer == bytesRead) {
                val read = inputStream.read(buffer, 0, bufferSize)
                if (read == -1) {
                    throw WavFileException("Not enough data available")
                }
                bytesRead = read
                bufferPointer = 0
            }
            var v = buffer[bufferPointer].toLong()
            if (b < format.bytesPerSample - 1 || format.bytesPerSample == 1) {
                v = v and 0xFF.toLong()
            }
            resultSample += (v shl b * 8)
            bufferPointer++
        }
        return resultSample
    }
}

/**
 * Read little-endian data from the buffer.
 *
 * @param buffer to read from
 * @param position the starting position to read from
 * @param count the number of bytes to read
 * @return a little-endian long value read from buffer
 */
internal fun readLittleEndian(buffer: ByteArray, position: Int, count: Int): Long {
    var currPosition = position + count - 1
    var returnValue = (buffer[currPosition].toLong() and 0xFF)
    for (b in 0 until count - 1) {
        returnValue = (returnValue shl 8) + (buffer[--currPosition].toLong() and 0xFF)
    }
    return returnValue
}

private fun readWavHeader(file: File, inputStream: InputStream, buffer: ByteArray) {
    val bytesRead = inputStream.read(buffer, 0, HEADER_LENGTH)
    if (bytesRead != HEADER_LENGTH) {
        throw WavFileException("Not enough wav file bytes for header")
    }
    val riffChunkID = readLittleEndian(buffer, 0, 4)
    val chunkSize = readLittleEndian(buffer, 4, 4)
    val riffTypeID = readLittleEndian(buffer, 8, 4)

    if (riffChunkID != RIFF_CHUNK_ID) {
        throw WavFileException("Invalid Wav Header data, incorrect riff chunk ID")
    }
    if (riffTypeID != RIFF_TYPE_ID) {
        throw WavFileException("Invalid Wav Header data, incorrect riff type ID")
    }
    if (file.length() < chunkSize + 8) {
        throw WavFileException("Header chunk size ($chunkSize) does not match file size (${file.length()})")
    }
}
