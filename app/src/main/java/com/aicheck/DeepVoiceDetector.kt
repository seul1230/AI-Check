package com.aicheck

import android.content.Context
import org.jtransforms.fft.FloatFFT_1D
import org.tensorflow.lite.Interpreter

import java.io.File
import kotlin.math.*

class DeepVoiceDetector(
    private val context: Context,
    private val interpreter: Interpreter
) {
    companion object {
        const val MAX_LEN = 400
        const val SAMPLE_RATE = 16000
        const val FRAME_LENGTH = 512
        const val FRAME_STEP = 256
        const val N_MELS = 128
        const val SEGMENT_DURATION = 4.0
        const val TRIM_MARGIN_SEC = 0.0
        const val IMG_HEIGHT = 128
        const val IMG_WIDTH = 500
    }

    fun detect(audioFilePath: String, threshold: Float = 0.4f): Map<String, Any?> {
        val waveform = Utils.decodeAudio(audioFilePath)
        val segments = splitIntoSegments(waveform)
        val segmentProbs = mutableListOf<Float>()
        val fullVector = getVectorFeatures(waveform)
        var fullSequence = getMelSequence(waveform)
        fullSequence = padSequence(fullSequence, MAX_LEN)
        val dummyImage = createDummyImage(IMG_WIDTH, IMG_HEIGHT)

        if (segments.isNotEmpty()) {
            segments.forEach { seg ->
                val segVector = getVectorFeatures(seg)
                var segSequence = getMelSequence(seg)
                if (segSequence.isEmpty()) return@forEach
                segSequence = padSequence(segSequence, MAX_LEN)
                val segSequenceInput = arrayOf(segSequence)
                val dummyImageInput = arrayOf(dummyImage)
                val segVectorInput = arrayOf(segVector)
                val inputArray = arrayOf(segSequenceInput, dummyImageInput, segVectorInput)
                val outputBuffer = Array(1) { FloatArray(2) }
                interpreter.runForMultipleInputsOutputs(inputArray, mapOf(0 to outputBuffer))
                segmentProbs.add(outputBuffer[0][1])
            }
        }

        val meanProb = if (segmentProbs.isNotEmpty()) segmentProbs.average().toFloat() else 0f
        val isDeepfakeSegment = if (segmentProbs.isNotEmpty()) meanProb > threshold else false

        val fullSequenceInput = arrayOf(fullSequence)
        val dummyImageInput = arrayOf(dummyImage)
        val fullVectorInput = arrayOf(fullVector)
        val fullInputArray = arrayOf(fullSequenceInput, dummyImageInput, fullVectorInput)
        val fullOutputBuffer = Array(1) { FloatArray(2) }
        interpreter.runForMultipleInputsOutputs(fullInputArray, mapOf(0 to fullOutputBuffer))
        val deepfakeProbFull = fullOutputBuffer[0][1]
        val isDeepfakeFull = deepfakeProbFull > threshold

        val basename = File(audioFilePath).name
        val trueLabel = if (basename.lowercase().contains("real")) "✅ Real" else "‼️ Fake or Mixed"

        return mapOf(
            "basename" to basename,
            "true_label" to trueLabel,
            "segment_probs" to segmentProbs,
            "mean_segment_prob" to meanProb,
            "is_deepfake_segment" to isDeepfakeSegment,
            "deepfake_prob_full" to deepfakeProbFull,
            "is_deepfake_full" to isDeepfakeFull
        )
    }

    private fun splitIntoSegments(waveform: FloatArray): List<FloatArray> {
        val segmentSamples = (SAMPLE_RATE * SEGMENT_DURATION).toInt()
        val marginSamples = (SAMPLE_RATE * TRIM_MARGIN_SEC).toInt()
        val totalSamples = waveform.size
        val trimmedWaveform = if (totalSamples > 3 * marginSamples) {
            waveform.copyOfRange(marginSamples, totalSamples - marginSamples)
        } else waveform
        val segments = mutableListOf<FloatArray>()
        var start = 0
        while (start < trimmedWaveform.size) {
            val end = min(start + segmentSamples, trimmedWaveform.size)
            val segment = trimmedWaveform.copyOfRange(start, end)
            if (segment.size >= FRAME_LENGTH) segments.add(segment)
            start += segmentSamples
        }
        return segments
    }

    private fun getVectorFeatures(waveform: FloatArray): FloatArray {
        val energy = waveform.map { it * it }.average().toFloat()
        var zeroCrossings = 0
        for (i in 0 until waveform.size - 1) if (waveform[i] * waveform[i + 1] < 0) zeroCrossings++
        val zcr = zeroCrossings.toFloat() / (waveform.size - 1)
        val maxAmp = waveform.maxOrNull()?.let { abs(it) } ?: 0f
        val meanAmp = waveform.average().toFloat()
        val stdAmp = sqrt(waveform.map { (it - meanAmp).pow(2) }.average()).toFloat()
        return floatArrayOf(energy, zcr, maxAmp, meanAmp, stdAmp) + FloatArray(5) { 0f }
    }

    private fun getMelSequence(waveform: FloatArray): Array<FloatArray> {
        val numFrames = ((waveform.size - FRAME_LENGTH) / FRAME_STEP) + 1
        if (numFrames <= 0) return emptyArray()
        val n_fft_bins = FRAME_LENGTH / 2 + 1
        val melFilterBanks = computeMelFilterBanks(N_MELS, n_fft_bins, SAMPLE_RATE, FRAME_LENGTH)
        val fft = FloatFFT_1D(FRAME_LENGTH.toLong())
        val melSpectrogram = Array(numFrames) { FloatArray(N_MELS) { 0f } }

        for (i in 0 until numFrames) {
            val start = i * FRAME_STEP
            val frame = if (start + FRAME_LENGTH <= waveform.size) {
                waveform.sliceArray(start until start + FRAME_LENGTH)
            } else {
                FloatArray(FRAME_LENGTH) { if (it + start < waveform.size) waveform[start + it] else 0f }
            }
            val windowedFrame = applyHanningWindow(frame)
            val fftData = windowedFrame.copyOf()
            fft.realForward(fftData)
            val magnitude = FloatArray(n_fft_bins)
            magnitude[0] = abs(fftData[0])
            for (k in 1 until FRAME_LENGTH / 2) {
                magnitude[k] = sqrt(fftData[2 * k].pow(2) + fftData[2 * k + 1].pow(2))
            }
            magnitude[FRAME_LENGTH / 2] = abs(fftData[1])
            for (m in 0 until N_MELS) {
                melSpectrogram[i][m] = ln(melFilterBanks[m].mapIndexed { k, v -> magnitude[k] * v }.sum() + 1e-6f)
            }
        }
        return melSpectrogram
    }

    private fun computeMelFilterBanks(n_mels: Int, n_fft_bins: Int, sampleRate: Int, frameLength: Int): Array<FloatArray> {
        val lowFreq = 0f
        val highFreq = sampleRate / 2f
        fun hzToMel(hz: Float): Float = 2595f * log10(1 + hz / 700f)
        fun melToHz(mel: Float): Float = 700f * (10f.pow(mel / 2595f) - 1)
        val lowMel = hzToMel(lowFreq)
        val highMel = hzToMel(highFreq)
        val melPoints = FloatArray(n_mels + 2) { lowMel + (highMel - lowMel) * it / (n_mels + 1) }
        val hzPoints = melPoints.map { melToHz(it) }.toFloatArray()
        val bin = hzPoints.map { floor((n_fft_bins - 1) * it / highFreq) }.toFloatArray()
        val filterBanks = Array(n_mels) { FloatArray(n_fft_bins) { 0f } }
        for (m in 1..n_mels) {
            val f_m_minus = bin[m - 1]
            val f_m = bin[m]
            val f_m_plus = bin[m + 1]
            for (k in 0 until n_fft_bins) {
                filterBanks[m - 1][k] = when {
                    k < f_m_minus -> 0f
                    k <= f_m -> (k - f_m_minus) / (f_m - f_m_minus + 1e-6f)
                    k <= f_m_plus -> (f_m_plus - k) / (f_m_plus - f_m + 1e-6f)
                    else -> 0f
                }
            }
        }
        return filterBanks
    }

    private fun applyHanningWindow(frame: FloatArray): FloatArray {
        val N = frame.size
        return FloatArray(N) { frame[it] * (0.5f * (1 - cos(2 * PI * it / (N - 1)).toFloat())) }
    }

    private fun padSequence(sequence: Array<FloatArray>, maxLen: Int): Array<FloatArray> {
        if (sequence.isEmpty()) return Array(maxLen) { FloatArray(N_MELS) { 0f } }
        val currentLen = sequence.size
        val n_mels = sequence[0].size
        return if (currentLen >= maxLen) {
            sequence.sliceArray(0 until maxLen)
        } else {
            Array(maxLen) { if (it < currentLen) sequence[it] else FloatArray(n_mels) { 0f } }
        }
    }

    private fun createDummyImage(width: Int, height: Int): Array<Array<FloatArray>> {
        return Array(height) { Array(width) { floatArrayOf(0f) } }
    }
}