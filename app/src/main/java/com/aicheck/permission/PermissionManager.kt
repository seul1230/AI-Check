package com.aicheck.permission

import android.Manifest
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.Settings
import android.util.Log
import androidx.activity.result.contract.ActivityResultContracts
import androidx.fragment.app.FragmentActivity

class PermissionManager(private val activity: FragmentActivity) {
    private val TAG = "PermissionManager"

    // ✅ 일반 권한 요청 (READ_PHONE_STATE, READ_CALL_LOG, RECORD_AUDIO 등)
    val requestPermissionsLauncher =
        activity.registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
            val phoneStateGranted = permissions[Manifest.permission.READ_PHONE_STATE] ?: false
            val callLogGranted = permissions[Manifest.permission.READ_CALL_LOG] ?: false

            if (phoneStateGranted && callLogGranted) {
                Log.d(TAG, "✅ 필수 권한 허용됨!")
                onPermissionsGranted?.invoke()
            } else {
                Log.e(TAG, "❌ 필수 권한 거부됨!")
                onPermissionsDenied?.invoke()
            }
        }

    // ✅ 저장소 관리 권한 요청 (Android 11 이상)
    private val requestManageStoragePermission =
        activity.registerForActivityResult(ActivityResultContracts.StartActivityForResult()) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                if (Environment.isExternalStorageManager()) {
                    Log.d(TAG, "✅ MANAGE_EXTERNAL_STORAGE 권한 허용됨!")
                } else {
                    Log.e(TAG, "❌ MANAGE_EXTERNAL_STORAGE 권한 거부됨!")
                }
            }
        }

    var onPermissionsGranted: (() -> Unit)? = null
    var onPermissionsDenied: (() -> Unit)? = null

    // ✅ 권한 요청 메서드
    fun requestPermissions(
        onGranted: () -> Unit,
        onDenied: () -> Unit
    ) {
        onPermissionsGranted = onGranted
        onPermissionsDenied = onDenied

        requestPermissionsLauncher.launch(
            arrayOf(
                Manifest.permission.READ_PHONE_STATE,
                Manifest.permission.READ_CALL_LOG,
                Manifest.permission.RECORD_AUDIO
            )
        )
    }

    // ✅ 저장소 권한 요청
    fun requestStoragePermission(function: () -> Unit) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            if (!Environment.isExternalStorageManager()) {
                val intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION)
                intent.data = Uri.parse("package:${activity.packageName}")
                requestManageStoragePermission.launch(intent)
            }
        } else {
            requestPermissionsLauncher.launch(
                arrayOf(
                    Manifest.permission.READ_EXTERNAL_STORAGE,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE
                )
            )
        }
    }
}
