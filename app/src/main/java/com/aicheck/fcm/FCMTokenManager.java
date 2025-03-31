package com.aicheck.fcm;

import android.util.Log;

import com.google.firebase.messaging.FirebaseMessaging;

public class FCMTokenManager {
    private static final String TAG = "FCMTokenManager";

    public interface TokenCallback {
        void onSuccess(String token);
        void onFailure(Exception e);
    }

    public static void getFCMToken(TokenCallback callback) {
        FirebaseMessaging.getInstance().getToken()
                .addOnCompleteListener(task -> {
                    if (task.isSuccessful()) {
                        String token = task.getResult();
                        Log.d(TAG, "✅ FCM 토큰 발급 성공: " + token);
                        callback.onSuccess(token);
                    } else {
                        Exception e = task.getException();
                        Log.e(TAG, "❌ FCM 토큰 발급 실패", e);
                        callback.onFailure(e != null ? e : new Exception("Unknown error"));
                    }
                });
    }
}