//package com.aicheck
//
//import ScalerParams
//import UrlDetector
//import android.content.Context
//import ai.onnxruntime.OrtEnvironment
//import ai.onnxruntime.OrtSession
//import kotlinx.serialization.json.Json
//import java.io.File
//
//object UrlModelManager {
//    lateinit var detector: UrlDetector
//
//    fun initialize(context: Context) {
//        val assetManager = context.assets
//
//        val onnxModelPath = copyModelToInternalStorage(context, "tfidf_1000_xgb_2000_2025_04_01_07_11_17_acc_9336_xgboost2.onnx")
//
//        val keywordListJson = assetManager.open("keyword_list.json").bufferedReader().use { it.readText() }
//        val vocabUrlJson = assetManager.open("vectorizer_vocab.json").bufferedReader().use { it.readText() }
//        val scalerParamsJson = assetManager.open("scaler_params.json").bufferedReader().use { it.readText() }
//
//        val keywordList = Json.decodeFromString<List<String>>(keywordListJson)
//        val vocab = Json.decodeFromString<Map<String, Int>>(vocabUrlJson)
//        val scalerParams = Json.decodeFromString<ScalerParams>(scalerParamsJson)
//
//        val ortEnv = OrtEnvironment.getEnvironment()
//        val ortSession = ortEnv.createSession(onnxModelPath, OrtSession.SessionOptions())
//
//        detector = UrlDetector(
//            ortSession = ortSession,
//            ortEnv = ortEnv,
//            keywordList = keywordList,
//            vocab = vocab,
//            scalerParams = scalerParams
//        )
//    }
//
//    private fun copyModelToInternalStorage(context: Context, modelFileName: String): String {
//        val outFile = File(context.filesDir, modelFileName)
//        if (!outFile.exists()) {
//            context.assets.open(modelFileName).use { input ->
//                outFile.outputStream().use { output ->
//                    input.copyTo(output)
//                }
//            }
//        }
//        return outFile.absolutePath
//    }
//}


// UrlModelManager
package com.aicheck

import android.content.Context
import android.util.Log
import com.chaquo.python.Python
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

object UrlModelManager {
    private lateinit var interpreter: Interpreter
    private lateinit var scalerJsonPath: String

    fun initialize(context: Context) {
        try {
            // 모델 로드
            interpreter = Interpreter(loadModelFile(context, "models/mlp-256-128-64-v6-auc-2-1_epoch_50_lr_0.001_batch_32_none_final_2025-04-09_00-17-44.tflite"))

            // scaler_params 저장
            scalerJsonPath = copyAssetToInternalStorage(context, "helpers/scaler_params_v6.json")

            Log.d("UrlModelManager", "✅ URL 모델 및 scaler 초기화 완료")
        } catch (e: Exception) {
            Log.e("UrlModelManager", "❌ 초기화 실패", e)
        }
    }

    fun detectUrl(context: Context, url: String): Float {
        try {
            val python = Python.getInstance()
            val pyModule = python.getModule("extractor_url")

            val result = pyModule.callAttr("extract_and_scale", url, scalerJsonPath)
            val featureVector = result.asList().map { it.toJava(Float::class.java) }.toFloatArray()
            Log.d("PythonVector", "features = ${featureVector.joinToString()}")

            val inputBuffer = ByteBuffer.allocateDirect(4 * featureVector.size).apply {
                order(ByteOrder.nativeOrder())
                featureVector.forEach { putFloat(it) }
                rewind()
            }

            val outputBuffer = ByteBuffer.allocateDirect(4).apply {
                order(ByteOrder.nativeOrder())
            }

            interpreter.run(inputBuffer, outputBuffer)
            outputBuffer.rewind()

            return outputBuffer.float
        } catch (e: Exception) {
            Log.e("UrlModelManager", "❌ URL 탐지 실패", e)
            return -1f
        }
    }

    private fun loadModelFile(context: Context, modelFileName: String): ByteBuffer {
        val fileDescriptor = context.assets.openFd(modelFileName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength).apply {
            order(ByteOrder.nativeOrder())
        }
    }

    private fun copyAssetToInternalStorage(context: Context, assetPath: String): String {
        val outFile = File(context.filesDir, assetPath)
        outFile.parentFile?.mkdirs()

        if (!outFile.exists()) {
            context.assets.open(assetPath).use { input ->
                outFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }
        return outFile.absolutePath
    }
}