package com.aicheck

import android.content.Context
import android.util.Log
import com.chaquo.python.Python
import com.chaquo.python.PyObject
import org.tensorflow.lite.Interpreter
import java.io.File
import kotlin.math.exp

class DeepVoiceDetectorWithChaquopy(
    private val context: Context,
    private val interpreter: Interpreter
) {
    companion object {
        const val IMG_HEIGHT = 128
        const val IMG_WIDTH = 500
        const val SEQ_LENGTH = 400
    }

    fun detect(audioFilePath: String, threshold: Float = 0.4f): Map<String, Any> {
        val py = Python.getInstance()
        val preprocess = py.getModule("preprocess_librosa")
        val audioFile = File(audioFilePath)
        val baseName = audioFile.nameWithoutExtension
        val melImagePath = File(context.cacheDir, "${baseName}_mel.png").absolutePath

        val result: PyObject = preprocess.callAttr(
            "process_audio_to_mel_and_features",
            audioFilePath,
            melImagePath
        )
        val resultMap = result.asMap()

        val melImageRaw = resultMap[PyObject.fromJava("mel_image")]!!
        val imageInput = Array(1) { Array(IMG_HEIGHT) { Array(IMG_WIDTH) { FloatArray(1) } } }
        for (y in 0 until IMG_HEIGHT) {
            for (x in 0 until IMG_WIDTH) {
                imageInput[0][y][x][0] = melImageRaw.asList()[y].asList()[x].asList()[0].toFloat()
            }
        }

        val fullSeq = resultMap[PyObject.fromJava("full_sequence")]!!.asList()
        val sequenceArray = Array(1) { Array(SEQ_LENGTH) { FloatArray(IMG_HEIGHT) } }
        for (i in fullSeq.indices) {
            val row = fullSeq[i].asList()
            sequenceArray[0][i] = row.map { it.toFloat() }.toFloatArray()
        }

        val fullVector = resultMap[PyObject.fromJava("full_vector")]!!
            .asList().map { it.toFloat() }.toFloatArray()
        val vectorArray = arrayOf(fullVector)

        Log.d("SHAPE", "📦 sequenceArray[0][0]: ${sequenceArray[0][0].joinToString()}")
        Log.d("SHAPE", "📦 vectorArray[0]: ${vectorArray[0].joinToString()}")
        val row0 = sequenceArray[0][0]
        Log.d("CHECK", "📦 Sequence[0] → max: ${row0.maxOrNull()}, min: ${row0.minOrNull()}, mean: ${row0.average()}")

        val output = Array(1) { FloatArray(2) }
        interpreter.runForMultipleInputsOutputs(
            arrayOf(sequenceArray, imageInput, vectorArray),
            mapOf(0 to output)
        )
        Log.d("DEBUG", "🔎 Full logits: ${output[0][0]}, ${output[0][1]}")
        val deepfakeProb = output[0][1]

        val segmentSequences = resultMap[PyObject.fromJava("segment_sequences")]?.asList()
        val segmentVectors = resultMap[PyObject.fromJava("segment_vectors")]?.asList()
        val segmentProbs = mutableListOf<Float>()

        if (segmentSequences != null && segmentVectors != null) {
            for (i in segmentSequences.indices) {
                val segSeqList = segmentSequences[i].asList()
                val segVecList = segmentVectors[i].asList().map { it.toFloat() }

                val segSeqArray = Array(1) { Array(SEQ_LENGTH) { FloatArray(IMG_HEIGHT) } }
                for (j in segSeqList.indices) {
                    val row = segSeqList[j].asList().map { it.toFloat() }
                    segSeqArray[0][j] = row.toFloatArray()
                }

                val segVecArray = arrayOf(segVecList.toFloatArray())
                val segOutput = Array(1) { FloatArray(2) }

                interpreter.runForMultipleInputsOutputs(
                    arrayOf(segSeqArray, imageInput, segVecArray),
                    mapOf(0 to segOutput)
                )

                Log.d("DEBUG", "🔎 Segment #$i logits: ${segOutput[0][0]}, ${segOutput[0][1]}")
                val softmaxSeg = softmax(segOutput[0])
                val segProb = softmaxSeg[1]
                segmentProbs.add(segProb)

                Log.d("DeepVoiceDetector", "📊 Segment #$i → $segProb")
            }
        }

        val meanSegmentProb = if (segmentProbs.isNotEmpty()) segmentProbs.average().toFloat() else 0f

        val isDeepfakeSegment = meanSegmentProb > threshold || isFlatFake(segmentProbs)
        val isDeepfakeFull = deepfakeProb > threshold
        val basename = File(audioFilePath).name

        return mapOf(
            "basename" to baseName,
            "true_label" to if (basename.lowercase().contains("real")) "✅ Real" else "‼️ Fake or Mixed",
            "segment_probs" to segmentProbs,
            "mean_segment_prob" to meanSegmentProb,
            "is_deepfake_segment" to isDeepfakeSegment,
            "deepfake_prob_full" to deepfakeProb,
            "is_deepfake_full" to isDeepfakeFull
        )
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0f
        val exps = logits.map { exp(it - maxLogit) }
        val sumExps = exps.sum()
        return exps.map { it / sumExps }.toFloatArray()
    }

    private fun isFlatFake(probs: List<Float>): Boolean {
        val mean = probs.average().toFloat()
        val max = probs.maxOrNull() ?: 0f

        if (probs.size == 1) {
            // 단일 세그먼트인 경우 → 확률이 일정 기준 이상이면 의심
            return max > 0.28f
        }

        val std = probs.map { (it - mean) * (it - mean) }.average().toFloat().let { Math.sqrt(it.toDouble()).toFloat() }
        return mean < 0.28f && std < 0.002f && max > 0.27f
    }


}
