package com.aicheck.ui

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import android.webkit.JavascriptInterface

class TokenWebInterface(context: Context) {
    private val sharedPreferences: SharedPreferences = context.getSharedPreferences("TokenStorage", Context.MODE_PRIVATE)
    private val TAG = "TokenWebInterface"

    @JavascriptInterface
    fun getAccessToken(): String {
        val accessToken = sharedPreferences.getString("accessToken", "No AccessToken")
        val refreshToken = sharedPreferences.getString("refreshToken", "No RefreshToken")
        val fcmToken = sharedPreferences.getString("fcmToken", "No FCM Token")

        return """{"accessToken": "$accessToken", "refreshToken": "$refreshToken", "fcmToken": "$fcmToken"}"""
    }

    @JavascriptInterface
    fun saveTokens(accessToken: String, refreshToken: String) {
        sharedPreferences.edit().apply {
            putString("accessToken", accessToken)
            putString("refreshToken", refreshToken)
            apply()
        }
        Log.d(TAG, "✅ AccessToken 및 RefreshToken 저장됨!")
    }

    @JavascriptInterface
    fun getFcmToken(): String? {
        val fcmToken = sharedPreferences.getString("fcmToken", null)
        Log.d(TAG, "📦 getFCMToken() 호출됨 → $fcmToken")
        return fcmToken
    }
}