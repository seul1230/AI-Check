//package com.example.myapplication
//
//import ai.onnxruntime.OnnxTensor
//import ai.onnxruntime.OrtEnvironment
//import ai.onnxruntime.OrtSession
//import android.net.Uri
//import kotlinx.serialization.Serializable
//import java.nio.FloatBuffer
//import java.util.Optional
//import kotlin.math.ln
//import ai.onnxruntime.OnnxValue
//@Serializable
//data class ScalerParams(val mean: List<Float>, val scale: List<Float>)
//
//class UrlDetector(
//    private val ortSession: OrtSession,
//    private val ortEnv: OrtEnvironment,
//    private val keywordList: List<String>,
//    private val vocab: Map<String, Int>,
//    private val scalerParams: ScalerParams
//) {
//    fun detect(url: String): String {
//        val domain = extractDomain(url)
//        val tfidf = vectorizeDomain(domain, vocab, 3000)
//        val features = extractFeatures(url, domain, keywordList)
//        val logScaled = applySelectiveLogScale(features, listOf(0, 3, 7, 8, 12, 13, 14, 15))
//        val normalized = applyStandardScaler(logScaled, scalerParams.mean.toFloatArray(), scalerParams.scale.toFloatArray())
//        val input = tfidf + normalized
//        val inputTensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(input), longArrayOf(1, input.size.toLong()))
//        val outputMap = ortSession.run(mapOf("input" to inputTensor))
//
//        // 출력 처리 수정
//        val probabilitiesValue = outputMap["probabilities"]
//        val probabilitiesTensor = when (probabilitiesValue) {
//            is OnnxTensor -> probabilitiesValue
//            is Optional<*> -> {
//                val optional = probabilitiesValue as Optional<OnnxValue>
//                if (optional.isPresent) {
//                    optional.get() as OnnxTensor
//                } else {
//                    throw IllegalStateException("모델 출력 값이 존재하지 않습니다.")
//                }
//            }
//            else -> throw IllegalStateException("예상치 못한 출력 타입: ${probabilitiesValue?.javaClass?.name}")
//        }
//
//        val probs = probabilitiesTensor.floatBuffer.array()
//        val result = if (probs[1] > probs[0]) "불법 URL" else "정상 URL"
//        return "검사 결과: $result\n(확률: ${probs.joinToString()})"
//    }
//
//    private fun extractDomain(url: String): String {
//        return Uri.parse(url).host ?: ""
//    }
//
//    private fun vectorizeDomain(domain: String, vocab: Map<String, Int>, maxFeatures: Int): FloatArray {
//        val padded = " $domain "
//        val vector = FloatArray(maxFeatures)
//        for (n in 3..5) {
//            for (i in 0..(padded.length - n)) {
//                val token = padded.substring(i, i + n)
//                vocab[token]?.let { if (it < maxFeatures) vector[it] = 1f }
//            }
//        }
//        return vector
//    }
//
//    private fun extractFeatures(url: String, domain: String, keywords: List<String>): FloatArray {
//        val cleanUrl = url.lowercase()
//        val features = mutableListOf<Float>()
//        fun maxSeq(regex: String) = Regex(regex).findAll(cleanUrl).map { it.value.length }.maxOrNull()?.toFloat() ?: 0f
//
//        features.add(cleanUrl.length.toFloat()) // 0: URL length
//        features.add(maxSeq("[a-z]+")) // 1: max lowercase sequence
//        features.add(maxSeq("[A-Z]+")) // 2: max uppercase sequence
//        features.add(maxSeq("\\d+")) // 3: max numeric sequence
//        features.add(if (url.length < 60) 1f else 0f) // 4: short URL
//        features.add(if (url.length > 100) 1f else 0f) // 5: long URL
//        val uri = Uri.parse(url)
//        val pathSegments = uri.pathSegments
//        features.add(pathSegments.getOrNull(0)?.length?.toFloat() ?: 0f) // 6: first path segment length
//        features.add(domain.length.toFloat()) // 7: domain length
//        features.add(keywords.count { cleanUrl.contains(it) }.toFloat()) // 8: keyword count
//        features.add(Regex("-").findAll(cleanUrl).count().toFloat()) // 9: hyphen count
//        features.add(Regex("=").findAll(cleanUrl).count().toFloat()) // 10: equal sign count
//        features.add(Regex("www").findAll(cleanUrl).count().toFloat()) // 11: "www" count
//        features.add(pathSegments.size.toFloat()) // 12: path depth
//        features.add(cleanUrl.count { !it.isLetterOrDigit() }.toFloat()) // 13: special character count
//        features.add(cleanUrl.count { it.isDigit() }.toFloat()) // 14: digit count
//        features.add(domain.split('.').size - 2f) // 15: subdomain count
//        features.add(cleanUrl.count { it.isLowerCase() }.toFloat()) // 16: lowercase letter count
//        features.add(cleanUrl.count { it.isUpperCase() }.toFloat()) // 17: uppercase letter count
//        features.add(domain.count { it.isLetter() }.toFloat()) // 18: domain letter count
//        features.add(domain.count { it.isDigit() }.toFloat()) // 19: domain digit count
//        features.add(domain.count { !it.isLetterOrDigit() }.toFloat()) // 20: domain special character count
//
//        return features.toFloatArray()
//    }
//
//    private fun applySelectiveLogScale(values: FloatArray, logIndices: List<Int>): FloatArray {
//        return FloatArray(values.size) { i ->
//            val v = values[i]
//            if (i in logIndices) ln(v + 1f) else v
//        }
//    }
//
//    private fun applyStandardScaler(values: FloatArray, mean: FloatArray, scale: FloatArray): FloatArray {
//        return FloatArray(values.size) { i -> (values[i] - mean[i]) / scale[i] }
//    }
//}

import ai.onnxruntime.*
import android.content.Context
import android.net.Uri
import kotlinx.serialization.*
import kotlinx.serialization.json.Json
import java.io.InputStreamReader
import java.nio.FloatBuffer
import java.util.Optional
import kotlin.math.ln

@Serializable
data class ScalerParams(val mean: List<Float>, val scale: List<Float>)

class UrlDetector(
    private val ortSession: OrtSession,
    private val ortEnv: OrtEnvironment,
    private val keywordList: List<String>,
    private val vocab: Map<String, Int>,
    private val scalerParams: ScalerParams
) {
    data class DetectionResult(
        val result: String,  // "정상 URL" or "불법 URL"
        val probability: List<Float>
    )

    fun detect(url: String): DetectionResult {
        val domain = extractDomain(url)
        val tfidf = vectorizeDomain(domain, vocab, 3000)
        val features = extractFeatures(url, domain, keywordList)
        val logScaled = applySelectiveLogScale(features, listOf(0, 3, 7, 8, 12, 13, 14, 15))
        val normalized = applyStandardScaler(logScaled, scalerParams.mean.toFloatArray(), scalerParams.scale.toFloatArray())
        val input = tfidf + normalized

        val inputTensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(input), longArrayOf(1, input.size.toLong()))
        val outputMap = ortSession.run(mapOf("input" to inputTensor))

        val probabilitiesValue = outputMap["probabilities"]
        val probabilitiesTensor = when (probabilitiesValue) {
            is OnnxTensor -> probabilitiesValue
            is Optional<*> -> {
                val optional = probabilitiesValue as Optional<OnnxValue>
                if (optional.isPresent) optional.get() as OnnxTensor
                else throw IllegalStateException("모델 출력 값이 존재하지 않습니다.")
            }
            else -> throw IllegalStateException("예상치 못한 출력 타입: ${probabilitiesValue?.javaClass?.name}")
        }

        val probs = probabilitiesTensor.floatBuffer.array()
        val result = if (probs[1] > probs[0]) "불법 URL" else "정상 URL"
        return DetectionResult(result, probs.toList())
    }

    private fun extractDomain(url: String): String {
        return Uri.parse(url).host ?: ""
    }

    private fun vectorizeDomain(domain: String, vocab: Map<String, Int>, maxFeatures: Int): FloatArray {
        val padded = " $domain "
        val vector = FloatArray(maxFeatures)
        for (n in 3..5) {
            for (i in 0..(padded.length - n)) {
                val token = padded.substring(i, i + n)
                vocab[token]?.let { if (it < maxFeatures) vector[it] = 1f }
            }
        }
        return vector
    }

    private fun extractFeatures(url: String, domain: String, keywords: List<String>): FloatArray {
        val cleanUrl = url.lowercase()
        val features = mutableListOf<Float>()
        fun maxSeq(regex: String) = Regex(regex).findAll(cleanUrl).map { it.value.length }.maxOrNull()?.toFloat() ?: 0f

        features.add(cleanUrl.length.toFloat())
        features.add(maxSeq("[a-z]+"))
        features.add(maxSeq("[A-Z]+"))
        features.add(maxSeq("\\d+"))
        features.add(if (url.length < 60) 1f else 0f)
        features.add(if (url.length > 100) 1f else 0f)
        val uri = Uri.parse(url)
        val pathSegments = uri.pathSegments
        features.add(pathSegments.getOrNull(0)?.length?.toFloat() ?: 0f)
        features.add(domain.length.toFloat())
        features.add(keywords.count { cleanUrl.contains(it) }.toFloat())
        features.add(Regex("-").findAll(cleanUrl).count().toFloat())
        features.add(Regex("=").findAll(cleanUrl).count().toFloat())
        features.add(Regex("www").findAll(cleanUrl).count().toFloat())
        features.add(pathSegments.size.toFloat())
        features.add(cleanUrl.count { !it.isLetterOrDigit() }.toFloat())
        features.add(cleanUrl.count { it.isDigit() }.toFloat())
        features.add(domain.split('.').size - 2f)
        features.add(cleanUrl.count { it.isLowerCase() }.toFloat())
        features.add(cleanUrl.count { it.isUpperCase() }.toFloat())
        features.add(domain.count { it.isLetter() }.toFloat())
        features.add(domain.count { it.isDigit() }.toFloat())
        features.add(domain.count { !it.isLetterOrDigit() }.toFloat())

        return features.toFloatArray()
    }

    private fun applySelectiveLogScale(values: FloatArray, logIndices: List<Int>): FloatArray {
        return FloatArray(values.size) { i ->
            val v = values[i]
            if (i in logIndices) ln(v + 1f) else v
        }
    }

    private fun applyStandardScaler(values: FloatArray, mean: FloatArray, scale: FloatArray): FloatArray {
        return FloatArray(values.size) { i -> (values[i] - mean[i]) / scale[i] }
    }

    companion object {
        inline fun <reified T> loadJsonFromAssets(context: Context, fileName: String): T {
            val jsonString = context.assets.open(fileName).use { InputStreamReader(it).readText() }
            return Json.decodeFromString(jsonString)
        }

        fun initialize(context: Context, modelPath: String): UrlDetector {
            val env = OrtEnvironment.getEnvironment()
            val session = env.createSession(modelPath, OrtSession.SessionOptions())
            val keywords = loadJsonFromAssets<List<String>>(context, "keyword_list.json")
            val vocab = loadJsonFromAssets<Map<String, Int>>(context, "vectorizer_vocab.json")
            val scaler = loadJsonFromAssets<ScalerParams>(context, "scaler_params.json")

            return UrlDetector(session, env, keywords, vocab, scaler)
        }
    }
}
