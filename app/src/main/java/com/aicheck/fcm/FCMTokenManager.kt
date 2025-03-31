package com.aicheck.fcm

import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.firebase.messaging.FirebaseMessaging

object FCMTokenManager {
    private const val TAG = "FCMTokenManager"

    fun getFCMToken(callback: TokenCallback) {
        FirebaseMessaging.getInstance().token
            .addOnCompleteListener { task: Task<String> ->
                if (task.isSuccessful) {
                    val token = task.result
                    Log.d(TAG, "✅ FCM 토큰 발급 성공: $token")
                    callback.onSuccess(token)
                } else {
                    val e = task.exception
                    Log.e(TAG, "❌ FCM 토큰 발급 실패", e)
                    callback.onFailure(e ?: Exception("Unknown error"))
                }
            }
    }

    interface TokenCallback {
        fun onSuccess(token: String?)
        fun onFailure(e: Exception?)
    }
}