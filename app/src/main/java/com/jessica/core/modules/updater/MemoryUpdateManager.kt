package com.jessica.core.modules.updater

import android.content.Context
import android.os.StatFs
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.file.Files

/**
 * Independent model memory update check (research-only).
 *
 * Fail-closed by design: we have ZERO approved signed on-device weight bundles
 * and ZERO on-device compatible model runtimes. This checker reports real
 * filesystem capacity without ever downloading experimental model weights.
 *
 * Approval catalog + signed releases + verified runtime are REQUIRED before
 * a download state or installation can be introduced. APK logic is untouched.
 */
internal class MemoryUpdateManager(context: Context) {
    private val appContext = context.applicationContext
    private val modelsRoot=File(appContext.filesDir, "calibri_models")
    private val _state=MutableStateFlow<MemoryUpdateState>(MemoryUpdateState.Idle)
    val state:StateFlow<MemoryUpdateState> = _state.asStateFlow()
    private var job: Job? = null

    fun checkForMemoryUpdates(scope:CoroutineScope) {
        if(job?.isActive==true)return
        job=scope.launch {
            _state.value=MemoryUpdateState.Checking
            try {
                val (free,used)=withContext(Dispatchers.IO) {
                    val stat=StatFs(appContext.filesDir.absolutePath)
                    val free=stat.availableBytes
                    val used=safeInstalledBytes(modelsRoot)
                    free to used
                }
                _state.value=MemoryUpdateState.NoApprovedModel(
                    filesystemFreeBytes=free,
                    appModelBytes=used,
                    budgetBytes=ModelReleaseGate.WEIGHTS_BUDGET_BYTES
                )
            } catch (t:Exception) {
                _state.value=MemoryUpdateState.Error(
                    t.message ?: "Не удалось проверить хранилище модели"
                )
            }
        }
    }

    companion object {
        /** Counts app-private regular files, rejecting symlinks and not following them. */
        private fun safeInstalledBytes(root:File):Long {
            if(!root.exists())return 0L
            if(Files.isSymbolicLink(root.toPath()) || !root.isDirectory)
                throw IllegalStateException("Unexpected model root type")
            var result=0L
            val stack=ArrayDeque<File>()
            stack.add(root)
            while(stack.isNotEmpty()){
                val f=stack.removeLast()
                val children=f.listFiles()
                    ?: throw IllegalStateException("Model directory inaccessible")
                for(child in children){
                    if(Files.isSymbolicLink(child.toPath()))
                        throw IllegalStateException("Unexpected symlink in model store")
                    if(child.isDirectory)stack.add(child)
                    else if(child.isFile){
                        val len=child.length()
                        if(len<0 || result>Long.MAX_VALUE-len)
                            throw IllegalStateException("Model store size overflow")
                        result+=len
                    } else throw IllegalStateException("Unexpected model store entry")
                }
            }
            return result
        }
    }
}
