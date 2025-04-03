package com.aicheck

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

object Utils {
    /** WAV 파일을 디코딩하여 FloatArray로 변환 (헤더 44바이트 건너뜀) */
    fun decodeAudio(filePath: String): FloatArray {
        val file = File(filePath)
        val bytes = file.readBytes()
        val dataBytes = bytes.copyOfRange(44, bytes.size)
        val numSamples = dataBytes.size / 2
        val samples = FloatArray(numSamples)
        val bb = ByteBuffer.wrap(dataBytes).order(ByteOrder.LITTLE_ENDIAN)
        for (i in 0 until numSamples) {
            samples[i] = bb.short.toFloat() / 32768.0f
        }
        return samples
    }

    /** 정수 시퀀스를 ByteBuffer로 변환 */
    fun convertSequenceToByteBuffer(sequence: List<Int>, maxLength: Int): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(4 * maxLength).order(ByteOrder.nativeOrder())
        for (value in sequence) {
            byteBuffer.putInt(value)
        }
        byteBuffer.rewind()
        return byteBuffer
    }

    /** 시퀀스 패딩 */
    fun padSequence(sequence: List<Int>, maxLength: Int): List<Int> {
        return if (sequence.size >= maxLength) {
            sequence.subList(0, maxLength)
        } else {
            sequence + List(maxLength - sequence.size) { 0 }
        }
    }
}