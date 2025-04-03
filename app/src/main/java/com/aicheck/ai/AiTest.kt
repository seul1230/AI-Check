package com.aicheck.ai

import android.content.Context
import android.util.Log
import org.jtransforms.fft.FloatFFT_1D
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.*

class AiTest(private val context: Context) {

    companion object {
        const val MAX_LEN = 400      // 모델이 기대하는 Mel-Spectrogram 시간 축 길이 (400)
        const val SAMPLE_RATE = 16000
        const val FRAME_LENGTH = 512
        const val FRAME_STEP = 256
        const val N_MELS = 128       // Mel 빈 수
        const val SEGMENT_DURATION = 4.0  // 세그먼트 길이 (초)
        const val TRIM_MARGIN_SEC = 0.0   // 여백 자르기 (초)
        const val IMG_HEIGHT = 128   // 모델이 기대하는 더미 이미지 높이
        const val IMG_WIDTH = 500    // 모델이 기대하는 더미 이미지 너비
    }

    /**
     * 멀티모달 TFLite 모델로 세그먼트 및 전체 파형 기반 추론 수행
     */
    fun predictMultimodalTfliteSegmented(
        audioFilePath: String,
        interpreter: Interpreter,
        threshold: Float = 0.4f
    ): Map<String, Any?> {
        // 1. 오디오 디코딩 IDEA및 전처리
        val waveform = decodeAudio(audioFilePath)
        val segments = splitIntoSegments(waveform)
        val segmentProbs = mutableListOf<Float>()

        val fullVector = getVectorFeatures(waveform)
        var fullSequence = getMelSequence(waveform)
        fullSequence = padSequence(fullSequence, MAX_LEN)
        // 더미 이미지는 3차원 배열로 생성하여 [height, width, 1] 모양을 만듭니다.
        val dummyImage = createDummyImage(IMG_WIDTH, IMG_HEIGHT)

        // 디버깅: 입력 형상 출력 (콘솔 출력)
        println("Full Mel-Spectrogram shape: [${fullSequence.size}, ${if(fullSequence.isNotEmpty()) fullSequence[0].size else "0"}]")
        println("Dummy Image shape: [${dummyImage.size}, ${if(dummyImage.isNotEmpty()) dummyImage[0].size else "0"}, ${if(dummyImage.isNotEmpty() && dummyImage[0].isNotEmpty()) dummyImage[0][0].size else "0"}]")
        println("Vector Features shape: [${fullVector.size}]")

        // interpreter의 입력/출력 텐서 정보 출력 (최신 API 사용)
        val inputCount = interpreter.inputTensorCount
        for (i in 0 until inputCount) {
            val tensor = interpreter.getInputTensor(i)
            Log.d("TFLite", "Input #$i: shape = ${tensor.shape().contentToString()}, type = ${tensor.dataType()}")
        }
        val outputCount = interpreter.outputTensorCount
        for (i in 0 until outputCount) {
            val tensor = interpreter.getOutputTensor(i)
            Log.d("TFLite", "Output #$i: shape = ${tensor.shape().contentToString()}, type = ${tensor.dataType()}")
        }

        // 2. 세그먼트 기반 추론
        if (segments.isNotEmpty()) {
            for (seg in segments) {
                val segVector = getVectorFeatures(seg)
                var segSequence = getMelSequence(seg)
                if (segSequence.isEmpty()) continue
                segSequence = padSequence(segSequence, MAX_LEN)

                // 입력 텐서는 배치 차원만 추가 (shape: [1, time, n_mels])
                val segSequenceInput = arrayOf(segSequence)       // [1, time, n_mels]
                // dummyImage는 이미 [128, 500, 1] 모양이므로 배치 차원 추가하면 [1, 128, 500, 1]
                val dummyImageInput = arrayOf(dummyImage)
                val segVectorInput = arrayOf(segVector)             // [1, 10]
                val inputArray: Array<Any> = arrayOf(segSequenceInput, dummyImageInput, segVectorInput)

                val outputBuffer = Array(1) { FloatArray(2) }
                val outputMap: Map<Int, Any> = mapOf(0 to outputBuffer)

                interpreter.runForMultipleInputsOutputs(inputArray, outputMap)
                segmentProbs.add(outputBuffer[0][1])
            }
        }

        val meanProb = if (segmentProbs.isNotEmpty()) segmentProbs.average().toFloat() else 0f
        val isDeepfakeSegment = if (segmentProbs.isNotEmpty()) {
            meanProb > threshold || isDeepfakeBySegments(segmentProbs)
        } else false

        // 3. 전체 파형 기반 추론
        val fullSequenceInput = arrayOf(fullSequence)         // [1, time, n_mels]
        val dummyImageInput = arrayOf(dummyImage)               // [1, 128, 500, 1]
        val fullVectorInput = arrayOf(fullVector)               // [1, 10]
        val fullInputArray: Array<Any> = arrayOf(fullSequenceInput, dummyImageInput, fullVectorInput)
        val fullOutputBuffer = Array(1) { FloatArray(2) }
        val fullOutputMap: Map<Int, Any> = mapOf(0 to fullOutputBuffer)

        interpreter.runForMultipleInputsOutputs(fullInputArray, fullOutputMap)
        val deepfakeProbFull: Float = fullOutputBuffer[0][1]
        val isDeepfakeFull = deepfakeProbFull > threshold

        val basename = File(audioFilePath).name
        val trueLabel = if (basename.lowercase().contains("real")) "✅ Real" else "‼️ Fake or Mixed"

        println("📞 [$basename]")
        println(">> True Label: $trueLabel")
        println("🎧 Full Waveform Deep Voice Probability: %.4f".format(deepfakeProbFull))
        println("🧠 Full Prediction: ${if (isDeepfakeFull) "‼️ Deepfake Voice Detected" else "✅ Real Voice"}")
        println("➡️ Segment-based Deep Voice Probabilities: ${segmentProbs.map { "%.4f".format(it) }}")
        if (segmentProbs.isNotEmpty()) {
            println("📊 Mean Deep Voice Probability: %.4f".format(meanProb))
            println("🧠 Segment-based / Final Prediction: ${if (isDeepfakeSegment) "‼️ Deepfake Voice Detected" else "✅ Real Voice"}")
        } else {
            println("📊 Segment-based Prediction: ❌ Skipped")
        }

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

    // --- 오디오 전처리 함수 ---

    /** WAV 파일을 디코딩하여 FloatArray로 반환 */
    private fun decodeAudio(filePath: String): FloatArray {
        val file = File(filePath)
        val bytes = file.readBytes()
        val dataBytes = bytes.copyOfRange(44, bytes.size) // WAV 헤더 건너뛰기
        val numSamples = dataBytes.size / 2
        val samples = FloatArray(numSamples)
        val bb = ByteBuffer.wrap(dataBytes).order(ByteOrder.LITTLE_ENDIAN)
        for (i in 0 until numSamples) {
            samples[i] = bb.short.toFloat() / 32768.0f // [-1, 1] 정규화
        }
        return samples
    }

    /** 파형을 세그먼트로 분할 (세그먼트는 최소 FRAME_LENGTH 이상) */
    private fun splitIntoSegments(waveform: FloatArray): List<FloatArray> {
        val segmentSamples = (SAMPLE_RATE * SEGMENT_DURATION).toInt()
        val marginSamples = (SAMPLE_RATE * TRIM_MARGIN_SEC).toInt()
        val totalSamples = waveform.size

        val trimmedWaveform = if (totalSamples > 3 * marginSamples) {
            waveform.copyOfRange(marginSamples, totalSamples - marginSamples)
        } else {
            waveform
        }

        val segments = mutableListOf<FloatArray>()
        var start = 0
        while (start < trimmedWaveform.size) {
            val end = min(start + segmentSamples, trimmedWaveform.size)
            val segment = trimmedWaveform.copyOfRange(start, end)
            // STFT 계산을 위해 최소 FRAME_LENGTH(512) 이상이어야 함
            if (segment.size >= FRAME_LENGTH) {
                segments.add(segment)
            }
            start += segmentSamples
        }
        return segments
    }

    /** 간단한 벡터 피처 추출 (에너지, ZCR 등) */
    private fun getVectorFeatures(waveform: FloatArray): FloatArray {
        val energy = waveform.map { it * it }.average().toFloat()
        var zeroCrossings = 0
        for (i in 0 until waveform.size - 1) {
            if (waveform[i] * waveform[i + 1] < 0) zeroCrossings++
        }
        val zcr = zeroCrossings.toFloat() / (waveform.size - 1)
        val maxAmp = waveform.maxOrNull()?.let { abs(it) } ?: 0f
        val meanAmp = waveform.average().toFloat()
        val stdAmp = sqrt(waveform.map { (it - meanAmp).pow(2) }.average()).toFloat()
        return floatArrayOf(energy, zcr, maxAmp, meanAmp, stdAmp) + FloatArray(5) { 0f } // 길이: 10
    }

    /** Mel-Spectrogram 계산 (출력 shape: [time, n_mels]) */
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
                val real = fftData[2 * k]
                val imag = fftData[2 * k + 1]
                magnitude[k] = sqrt(real * real + imag * imag)
            }
            magnitude[FRAME_LENGTH / 2] = abs(fftData[1])

            for (m in 0 until N_MELS) {
                var sum = 0f
                for (k in 0 until n_fft_bins) {
                    sum += magnitude[k] * melFilterBanks[m][k]
                }
                melSpectrogram[i][m] = ln(sum + 1e-6f)
            }
        }
        return melSpectrogram
    }

    /** Mel 필터 뱅크 생성 */
    private fun computeMelFilterBanks(n_mels: Int, n_fft_bins: Int, sampleRate: Int, frameLength: Int): Array<FloatArray> {
        val lowFreq = 0f
        val highFreq = sampleRate / 2f

        fun hzToMel(hz: Float): Float = 2595f * log10(1 + hz / 700f)
        fun melToHz(mel: Float): Float = 700f * (10f.pow(mel / 2595f) - 1)

        val lowMel = hzToMel(lowFreq)
        val highMel = hzToMel(highFreq)
        val melPoints = FloatArray(n_mels + 2) { i -> lowMel + (highMel - lowMel) * i / (n_mels + 1) }
        val hzPoints = FloatArray(n_mels + 2) { i -> melToHz(melPoints[i]) }
        val bin = FloatArray(n_mels + 2) { i -> floor((n_fft_bins - 1) * hzPoints[i] / highFreq) }
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

    /** Hanning 창 적용 */
    private fun applyHanningWindow(frame: FloatArray): FloatArray {
        val N = frame.size
        return FloatArray(N) { n -> frame[n] * (0.5f * (1 - cos(2 * Math.PI * n / (N - 1)).toFloat())) }
    }

    /** 시퀀스 패딩 또는 트리밍 */
    private fun padSequence(sequence: Array<FloatArray>, maxLen: Int): Array<FloatArray> {
        if (sequence.isEmpty()) return Array(maxLen) { FloatArray(N_MELS) { 0f } }
        val currentLen = sequence.size
        val n_mels = sequence[0].size
        return if (currentLen >= maxLen) {
            sequence.sliceArray(0 until maxLen)
        } else {
            val padded = Array(maxLen) { FloatArray(n_mels) { 0f } }
            for (i in 0 until currentLen) padded[i] = sequence[i]
            padded
        }
    }

    // --- 더미 이미지 생성 ---
    /**
     * 더미 이미지 생성 (채널 추가)
     * 출력 모양: [height, width, 1]
     */
    private fun createDummyImage(width: Int, height: Int): Array<Array<FloatArray>> {
        return Array(height) { Array(width) { floatArrayOf(0f) } }
    }

    // --- 후처리 및 판단 함수 ---
    private fun isDeepfakeBySegments(probs: List<Float>): Boolean {
        val mean = probs.average().toFloat()
        val std = sqrt(probs.map { (it - mean).pow(2) }.average()).toFloat()
        return (mean > 0.4f || probs.count { it > 0.9f } >= 2 || isFlatFake(probs))
    }

    private fun isFlatFake(probList: List<Float>): Boolean {
        val mean = probList.average().toFloat()
        val std = sqrt(probList.map { (it - mean).pow(2) }.average()).toFloat()
        val maxP = probList.maxOrNull() ?: 0f
        return (mean < 0.01f && std < 0.003f && maxP > 0.005f)
    }
}
