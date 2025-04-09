package com.aicheck

import ScalerParams
import UrlDetector
import android.content.Context
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.serialization.json.Json
import java.io.File

object UrlModelManager {
    lateinit var detector: UrlDetector

    fun initialize(context: Context) {
        val assetManager = context.assets

        val onnxModelPath = copyModelToInternalStorage(context, "tfidf_1000_xgb_2000_2025_04_01_07_11_17_acc_9336_xgboost2.onnx")

        val keywordListJson = assetManager.open("keyword_list.json").bufferedReader().use { it.readText() }
        val vocabUrlJson = assetManager.open("vectorizer_vocab.json").bufferedReader().use { it.readText() }
        val scalerParamsJson = assetManager.open("scaler_params.json").bufferedReader().use { it.readText() }

        val keywordList = Json.decodeFromString<List<String>>(keywordListJson)
        val vocab = Json.decodeFromString<Map<String, Int>>(vocabUrlJson)
        val scalerParams = Json.decodeFromString<ScalerParams>(scalerParamsJson)

        val ortEnv = OrtEnvironment.getEnvironment()
        val ortSession = ortEnv.createSession(onnxModelPath, OrtSession.SessionOptions())

        detector = UrlDetector(
            ortSession = ortSession,
            ortEnv = ortEnv,
            keywordList = keywordList,
            vocab = vocab,
            scalerParams = scalerParams
        )
    }

    private fun copyModelToInternalStorage(context: Context, modelFileName: String): String {
        val outFile = File(context.filesDir, modelFileName)
        if (!outFile.exists()) {
            context.assets.open(modelFileName).use { input ->
                outFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }
        return outFile.absolutePath
    }
}
