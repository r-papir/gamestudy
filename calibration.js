/**
 * Eye Tracking Calibration Module
 * Handles WebGazer initialization and 9-point click calibration
 */

const Calibration = (function() {
    // Constants
    const CLICKS_PER_POINT = 5;
    const TOTAL_POINTS = 9;
    const CALIBRATION_FLAG_KEY = 'webgazerCalibrationComplete';
    const CALIBRATION_SKIPPED_KEY = 'eyeTrackingSkipped';

    // State
    let pointsCompleted = 0;
    let webgazerReady = false;

    /**
     * Check if calibration data exists in sessionStorage
     */
    function hasExistingCalibration() {
        const calibrationFlag = sessionStorage.getItem(CALIBRATION_FLAG_KEY);
        return calibrationFlag === 'true';
    }

    /**
     * Mark calibration as complete
     */
    function setCalibrationComplete() {
        sessionStorage.setItem(CALIBRATION_FLAG_KEY, 'true');
    }

    /**
     * Clear calibration data for recalibration
     */
    function clearCalibration() {
        sessionStorage.removeItem(CALIBRATION_FLAG_KEY);
        sessionStorage.removeItem(CALIBRATION_SKIPPED_KEY);
        localStorage.removeItem('webgazerGlobalData');
        if (webgazerReady && typeof webgazer !== 'undefined') {
            webgazer.clearData();
        }
    }

    /**
     * Skip calibration and continue without eye tracking
     */
    function skipCalibration() {
        sessionStorage.setItem(CALIBRATION_SKIPPED_KEY, 'true');
        window.location.href = 'index.html';
    }

    /**
     * Show specific phase, hide others
     */
    function showPhase(phaseId) {
        document.querySelectorAll('.phase').forEach(phase => {
            phase.classList.add('hidden');
        });
        document.getElementById(phaseId).classList.remove('hidden');
    }

    /**
     * Update progress bar
     */
    function updateProgress() {
        const percentage = (pointsCompleted / TOTAL_POINTS) * 100;
        document.getElementById('progress-fill').style.width = percentage + '%';
        document.getElementById('progress-text').textContent =
            pointsCompleted + ' / ' + TOTAL_POINTS + ' points';
    }

    /**
     * Initialize WebGazer and request camera permission
     */
    async function initWebGazer() {
        if (typeof webgazer === 'undefined') {
            throw new Error('WebGazer library not loaded. Please check your internet connection.');
        }

        try {
            // Clear any potentially corrupted localStorage data first
            // This prevents "t is not a function" errors from loading bad data
            localStorage.removeItem('webgazerGlobalData');

            // Initialize WebGazer
            await webgazer.begin();

            // Configure AFTER begin() to avoid "t is not a function" errors
            // from calling methods before internal state is initialized
            webgazer.saveDataAcrossSessions(false);

            // Show video preview during calibration for user feedback
            webgazer.showVideoPreview(true);
            webgazer.showPredictionPoints(true);

            webgazerReady = true;
            return true;
        } catch (error) {
            console.error('WebGazer init failed:', error);
            throw error;
        }
    }

    /**
     * Handle calibration point click
     */
    function handlePointClick(event) {
        const point = event.target;
        if (point.classList.contains('complete')) return;

        let clicks = parseInt(point.dataset.clicks) || 0;
        clicks++;
        point.dataset.clicks = clicks;

        if (clicks >= CLICKS_PER_POINT) {
            point.classList.add('complete');
            pointsCompleted++;
            updateProgress();

            // Show center point after 8 outer points complete
            if (pointsCompleted === 8) {
                const centerPoint = document.getElementById('pt5');
                centerPoint.style.display = 'block';
            }

            // Check completion
            if (pointsCompleted >= TOTAL_POINTS) {
                completeCalibration();
            }
        }
    }

    /**
     * Complete calibration and transition to next phase
     */
    function completeCalibration() {
        // Hide video preview and prediction points for cleaner game experience
        if (webgazerReady && typeof webgazer !== 'undefined') {
            webgazer.showVideoPreview(false);
            webgazer.showPredictionPoints(false);
        }

        // Mark calibration complete
        setCalibrationComplete();

        // Transition to completion phase
        showPhase('completion-phase');
    }

    /**
     * Reset for recalibration
     */
    function recalibrate() {
        clearCalibration();
        pointsCompleted = 0;

        // Reset all points
        document.querySelectorAll('.calibration-point').forEach(function(point) {
            point.dataset.clicks = '0';
            point.classList.remove('complete');
        });

        // Hide center point again
        document.getElementById('pt5').style.display = 'none';
        updateProgress();

        // Show video preview again
        if (webgazerReady && typeof webgazer !== 'undefined') {
            webgazer.showVideoPreview(true);
            webgazer.showPredictionPoints(true);
        }

        showPhase('calibration-phase');
    }

    /**
     * Navigate to games (or return URL if specified)
     */
    function continueToGames() {
        // Pause WebGazer to force it to save data to localStorage before navigating
        if (webgazerReady && typeof webgazer !== 'undefined') {
            webgazer.pause();
        }

        // Ensure calibration flag is set before navigating (defensive - may already be set)
        setCalibrationComplete();

        // Small delay to ensure all data is saved
        setTimeout(function() {
            // Verify flag is set, retry if needed
            if (sessionStorage.getItem(CALIBRATION_FLAG_KEY) !== 'true') {
                sessionStorage.setItem(CALIBRATION_FLAG_KEY, 'true');
            }

            const params = new URLSearchParams(window.location.search);
            const returnUrl = params.get('return');
            if (returnUrl && returnUrl.startsWith('/')) {
                window.location.href = returnUrl;
            } else {
                window.location.href = 'index.html';
            }
        }, 200);
    }

    /**
     * Initialize calibration page
     */
    async function init() {
        const params = new URLSearchParams(window.location.search);
        const forceRecalibration = params.get('force') === 'true';

        // Check if already calibrated and not forcing recalibration
        if (hasExistingCalibration() && !forceRecalibration) {
            window.location.href = 'index.html';
            return;
        }

        // If forcing recalibration, clear old data
        if (forceRecalibration) {
            clearCalibration();
        }

        // Setup calibration point click listeners
        document.querySelectorAll('.calibration-point').forEach(function(point) {
            point.addEventListener('click', handlePointClick);
        });

        // Hide center point initially
        document.getElementById('pt5').style.display = 'none';

        // Setup button listeners
        document.getElementById('start-calibration-btn').addEventListener('click', function() {
            showPhase('calibration-phase');
        });

        document.getElementById('continue-btn').addEventListener('click', continueToGames);

        document.getElementById('recalibrate-btn').addEventListener('click', recalibrate);

        document.getElementById('retry-btn').addEventListener('click', function() {
            location.reload();
        });

        // Check for camera/WebGazer support
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            document.getElementById('error-message').textContent =
                'Your browser does not support camera access. Please use a modern browser like Chrome or Firefox.';
            showPhase('error-phase');
            return;
        }

        // Initialize WebGazer
        try {
            await initWebGazer();
            document.getElementById('camera-status').textContent = 'Camera ready!';
            document.getElementById('camera-status').classList.remove('error');
            document.getElementById('camera-status').classList.add('success');
            document.getElementById('start-calibration-btn').disabled = false;
        } catch (error) {
            let errorMsg = 'Failed to access camera. ';
            if (error.name === 'NotAllowedError') {
                errorMsg += 'Camera permission was denied. Please allow camera access and try again.';
            } else if (error.name === 'NotFoundError') {
                errorMsg += 'No camera found on this device.';
            } else {
                errorMsg += error.message || 'Unknown error occurred.';
            }
            document.getElementById('error-message').textContent = errorMsg;
            showPhase('error-phase');
        }
    }

    // Public API
    return {
        init: init,
        hasExistingCalibration: hasExistingCalibration,
        clearCalibration: clearCalibration,
        recalibrate: recalibrate,
        skipCalibration: skipCalibration
    };
})();

// Auto-initialize on page load
document.addEventListener('DOMContentLoaded', Calibration.init);
