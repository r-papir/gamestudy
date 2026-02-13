/**
 * S3 Upload Module for Game Study
 * Handles uploading session data to AWS S3 via Lambda
 *
 * Usage:
 * 1. Include this script: <script src="s3-uploader.js"></script>
 * 2. Set the API URL: S3Uploader.setApiUrl('https://your-api-gateway-url.amazonaws.com/prod')
 * 3. Call S3Uploader.upload(...) when game completes
 */

const S3Uploader = (function() {
    // Configuration
    const config = {
        apiBaseUrl: '', // Set via S3Uploader.setApiUrl()
        maxRetries: 3,
        retryDelayMs: 1000
    };

    // Upload state
    let uploadInProgress = false;
    let uploadStatus = 'idle'; // 'idle', 'uploading', 'success', 'error'

    // Backup state
    let backupInterval = null;
    let backupSession = null; // { sessionCode, gameName, basePath, audioUploadUrl, audioKey, getAudioChunks }

    /**
     * Initialize upload session - gets presigned URLs from Lambda
     */
    async function initUpload(sessionCode, gameName) {
        const response = await fetch(`${config.apiBaseUrl}/init-upload`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sessionCode, gameName })
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Init upload failed: ${response.status} - ${errorText}`);
        }

        return response.json();
    }

    /**
     * Upload audio directly to S3 via presigned URL (XHR for progress)
     * @param {string} presignedUrl - S3 presigned PUT URL
     * @param {Blob} audioBlob - Audio data to upload
     * @param {function} onProgress - Optional callback: (percent) => void
     */
    function uploadAudio(presignedUrl, audioBlob, onProgress) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            xhr.open('PUT', presignedUrl);
            xhr.setRequestHeader('Content-Type', 'audio/webm');

            if (onProgress) {
                xhr.upload.onprogress = (e) => {
                    if (e.lengthComputable) {
                        const pct = Math.round((e.loaded / e.total) * 100);
                        onProgress(pct);
                    }
                };
            }

            xhr.onload = () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    resolve(true);
                } else {
                    reject(new Error(`Audio upload failed: ${xhr.status}`));
                }
            };

            xhr.onerror = () => reject(new Error('Audio upload network error'));
            xhr.send(audioBlob);
        });
    }

    /**
     * Submit session data to Lambda
     */
    async function submitSessionData(basePath, sessionData, eyeTrackingData, audioKey) {
        const response = await fetch(`${config.apiBaseUrl}/submit-session`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                basePath,
                sessionData,
                eyeTrackingData,
                audioKey
            })
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Session submit failed: ${response.status} - ${errorText}`);
        }

        return response.json();
    }

    /**
     * Main upload function - orchestrates the full upload flow
     * @param {string} sessionCode - The session code (e.g., "1234")
     * @param {string} gameName - The game name (e.g., "Game A")
     * @param {object} sessionData - The session data object with movements, etc.
     * @param {object} eyeTrackingData - Eye tracking data with gaze array
     * @param {Array} audioChunks - Array of audio Blob chunks from MediaRecorder
     * @param {function} onProgress - Optional callback for progress updates
     */
    async function uploadSession(sessionCode, gameName, sessionData, eyeTrackingData, audioChunks, onProgress) {
        if (uploadInProgress) {
            console.warn('S3Uploader: Upload already in progress');
            return { success: false, error: 'Upload already in progress' };
        }

        if (!config.apiBaseUrl) {
            console.error('S3Uploader: API URL not configured. Call S3Uploader.setApiUrl() first.');
            return { success: false, error: 'API URL not configured' };
        }

        uploadInProgress = true;
        uploadStatus = 'uploading';

        try {
            // Step 1: Initialize upload session (reuse backup session if available and valid)
            let basePath, audioUploadUrl, audioKey;
            if (backupSession && backupSession.basePath && backupSession.audioUploadUrl) {
                basePath = backupSession.basePath;
                audioUploadUrl = backupSession.audioUploadUrl;
                audioKey = backupSession.audioKey;
                onProgress && onProgress('Reusing backup session...');
                console.log('S3Uploader: Reusing backup session', basePath);
            } else {
                onProgress && onProgress('Initializing upload...');
                console.log('S3Uploader: Creating new upload session');
                const initResult = await initUpload(sessionCode, gameName);
                basePath = initResult.basePath;
                audioUploadUrl = initResult.audioUploadUrl;
                audioKey = initResult.audioKey;
            }

            // Step 2: Upload audio if available
            if (audioChunks && audioChunks.length > 0) {
                onProgress && onProgress('Uploading audio (0%)...');
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                await uploadAudio(audioUploadUrl, audioBlob, (pct) => {
                    onProgress && onProgress(`Uploading audio (${pct}%)...`);
                });
            }

            // Step 3: Submit session data
            onProgress && onProgress('Uploading session data...');
            const result = await submitSessionData(basePath, sessionData, eyeTrackingData, audioKey);

            uploadStatus = 'success';
            onProgress && onProgress('Upload complete!');

            return { success: true, ...result };

        } catch (error) {
            console.error('S3Uploader: Upload failed:', error);
            uploadStatus = 'error';
            onProgress && onProgress(`Upload failed: ${error.message}`);

            return { success: false, error: error.message };

        } finally {
            uploadInProgress = false;
        }
    }

    /**
     * Upload with retry logic and exponential backoff
     */
    async function uploadWithRetry(sessionCode, gameName, sessionData, eyeTrackingData, audioChunks, onProgress) {
        let lastError;

        for (let attempt = 1; attempt <= config.maxRetries; attempt++) {
            try {
                const result = await uploadSession(sessionCode, gameName, sessionData, eyeTrackingData, audioChunks, onProgress);
                if (result.success) {
                    return result;
                }
                lastError = new Error(result.error);
            } catch (error) {
                lastError = error;
            }

            if (attempt < config.maxRetries) {
                const delay = config.retryDelayMs * Math.pow(2, attempt - 1);
                onProgress && onProgress(`Retry ${attempt}/${config.maxRetries} in ${delay / 1000}s...`);
                await new Promise(resolve => setTimeout(resolve, delay));
            }
        }

        return { success: false, error: lastError?.message || 'Upload failed after retries' };
    }

    /**
     * Start periodic audio backups during recording
     * @param {string} sessionCode - The session code
     * @param {string} gameName - The game name
     * @param {function} getAudioChunks - Function that returns current audio chunks array
     * @param {number} intervalMs - Backup interval in ms (default 5 min)
     */
    async function startBackups(sessionCode, gameName, getAudioChunks, intervalMs = 300000) {
        if (!config.apiBaseUrl) {
            console.warn('S3Uploader: API URL not configured, skipping backups.');
            return;
        }

        try {
            const initResult = await initUpload(sessionCode, gameName);
            backupSession = {
                sessionCode,
                gameName,
                basePath: initResult.basePath,
                audioUploadUrl: initResult.audioUploadUrl,
                audioKey: initResult.audioKey,
                getAudioChunks
            };
            console.log('S3Uploader: Backup session initialized', backupSession.basePath);

            backupInterval = setInterval(async () => {
                const chunks = backupSession.getAudioChunks();
                if (!chunks || chunks.length === 0) return;

                try {
                    const blob = new Blob(chunks, { type: 'audio/webm' });
                    await uploadAudio(backupSession.audioUploadUrl, blob);
                    console.log(`S3Uploader: Backup uploaded (${(blob.size / 1024 / 1024).toFixed(1)} MB)`);
                } catch (err) {
                    console.warn('S3Uploader: Backup upload failed:', err.message);
                }
            }, intervalMs);
        } catch (err) {
            console.warn('S3Uploader: Failed to init backup session:', err.message);
        }
    }

    /**
     * Stop periodic audio backups
     */
    function stopBackups() {
        if (backupInterval) {
            clearInterval(backupInterval);
            backupInterval = null;
        }
        // Don't clear backupSession — uploadSession() will reuse it for the final upload
    }

    // Public API
    return {
        /**
         * Upload session data to S3 via Lambda
         */
        upload: uploadWithRetry,

        /**
         * Get current upload status
         */
        getStatus: () => uploadStatus,

        /**
         * Check if upload is in progress
         */
        isUploading: () => uploadInProgress,

        /**
         * Set the API Gateway base URL (required before uploading)
         * @param {string} url - e.g., 'https://abc123.execute-api.us-east-1.amazonaws.com/prod'
         */
        setApiUrl: (url) => {
            config.apiBaseUrl = url.replace(/\/$/, ''); // Remove trailing slash
        },

        /**
         * Get the configured API URL
         */
        getApiUrl: () => config.apiBaseUrl,

        /**
         * Start periodic audio backups during recording
         */
        startBackups,

        /**
         * Stop periodic audio backups
         */
        stopBackups
    };
})();
