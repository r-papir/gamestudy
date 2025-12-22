/**
 * Audio Recording Module for Game Study
 * Provides audio recording with speech transcription and eye tracking functionality
 *
 * Usage:
 * 1. Include WebGazer: <script src="https://webgazer.cs.brown.edu/webgazer.js"></script>
 * 2. Include this script: <script src="audio-recorder.js"></script>
 * 3. Call AudioRecorder.init(config) with your game-specific configuration
 * 4. The recording UI will be automatically added to the page
 */

const AudioRecorder = (function() {
    // Recording state
    const state = {
        isRecording: false,
        startTime: null,
        keystrokes: [],
        audioData: null,
        transcription: [],
        mediaRecorder: null,
        recognition: null,
        audioChunks: [],
        gameId: null,
        actionCounter: 0,
        gazeData: [],           // Array of [x, y, timestamp] tuples
        webgazerStarted: false
    };

    // Configuration - can be overridden by init()
    let config = {
        gamePrefix: 'game',
        getGameState: () => ({}),  // Function to capture current game state
        onKeystroke: null,         // Optional callback when keystroke is recorded
        screenToGrid: null,        // Function to convert screen (x,y) to grid coords, returns {x, y} or null if off-grid
        keyActionMap: {},          // Map of key names to semantic action names (e.g., {'ArrowUp': 'move_chunk_up'})
        insertBeforeSelector: null, // CSS selector - insert UI before this element (e.g., '#download-button')
    };

    // DOM elements
    let elements = {
        recordBtn: null,
        statusDiv: null,
        container: null
    };

    // Generate unique IDs
    function generateGameId() {
        const timestamp = Date.now().toString(36);
        const random = Math.random().toString(36).substr(2, 8);
        return `gs-${config.gamePrefix}-${timestamp}${random}`;
    }

    function generateGuid() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0;
            const v = c === 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }

    // Initialize WebGazer
    async function initWebGazer() {
        if (typeof webgazer === 'undefined') {
            console.warn('WebGazer not loaded. Eye tracking will be disabled.');
            return false;
        }

        try {
            await webgazer
                .setGazeListener((data, timestamp) => {
                    if (data && state.isRecording) {
                        const relativeTimestamp = Date.now() - state.startTime;

                        // Convert to grid coordinates if screenToGrid function provided
                        if (config.screenToGrid) {
                            const gridPos = config.screenToGrid(data.x, data.y);
                            if (gridPos) {
                                // On grid: store as [gridX, gridY, timestamp]
                                state.gazeData.push([gridPos.x, gridPos.y, relativeTimestamp]);
                            } else {
                                // Off grid: store as [null, null, timestamp]
                                state.gazeData.push([null, null, relativeTimestamp]);
                            }
                        } else {
                            // No conversion function, store raw screen coords
                            state.gazeData.push([
                                Math.round(data.x),
                                Math.round(data.y),
                                relativeTimestamp
                            ]);
                        }
                    }
                })
                .saveDataAcrossSessions(true)
                .begin();

            // Hide video preview and prediction points for cleaner UI
            webgazer.showVideoPreview(false).showPredictionPoints(false);
            state.webgazerStarted = true;
            console.log('WebGazer initialized successfully');
            return true;
        } catch (error) {
            console.error('Failed to initialize WebGazer:', error);
            return false;
        }
    }

    // Pause/resume WebGazer based on recording state
    function setWebGazerActive(active) {
        if (!state.webgazerStarted || typeof webgazer === 'undefined') return;

        if (active) {
            webgazer.resume();
        } else {
            webgazer.pause();
        }
    }

    // Find nearby speech for reasoning context
    function findNearbyReasoning(actionTimestamp) {
        const speechBefore = state.transcription.filter(t =>
            t.timestamp <= actionTimestamp &&
            (actionTimestamp - t.timestamp) <= 3000
        );

        const speechAfter = state.transcription.filter(t =>
            t.timestamp > actionTimestamp &&
            (t.timestamp - actionTimestamp) <= 1000
        );

        const allSpeech = [...speechBefore, ...speechAfter]
            .sort((a, b) => Math.abs(a.timestamp - actionTimestamp) - Math.abs(b.timestamp - actionTimestamp));

        return allSpeech.length > 0 ? allSpeech[0].text : "No speech detected";
    }

    // Record a keystroke event
    function recordKeystroke(key, action, timestamp) {
        if (!state.isRecording) return;

        const frame = config.getGameState();

        // Keep simple format for export
        state.keystrokes.push({
            key: key,
            action: action,
            timestamp: timestamp,
            gameState: frame
        });

        if (config.onKeystroke) {
            config.onKeystroke(key, action, timestamp);
        }
    }

    // Start recording
    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

            // Setup MediaRecorder for audio
            state.mediaRecorder = new MediaRecorder(stream);
            state.audioChunks = [];

            state.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    state.audioChunks.push(event.data);
                }
            };

            // Setup Web Speech API for transcription
            if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                state.recognition = new SpeechRecognition();
                state.recognition.continuous = true;
                state.recognition.interimResults = true;
                state.recognition.lang = 'en-US';

                state.recognition.onstart = () => {
                    updateStatusMessage('Speech recognition active');
                };

                state.recognition.onresult = (event) => {
                    for (let i = event.resultIndex; i < event.results.length; i++) {
                        const transcript = event.results[i][0].transcript;
                        const timestamp = Date.now() - state.startTime;
                        const isFinal = event.results[i].isFinal;
                        const confidence = event.results[i][0].confidence;

                        if (isFinal) {
                            state.transcription.push({
                                text: transcript,
                                timestamp: timestamp,
                                confidence: confidence
                            });
                            updateStatusMessage(`Transcribed: "${transcript.substring(0, 30)}${transcript.length > 30 ? '...' : ''}"`);
                        }
                    }
                };

                state.recognition.onerror = (event) => {
                    console.error('Speech recognition error:', event.error);
                    updateStatusMessage(`Speech error: ${event.error}`);
                };

                state.recognition.onend = () => {
                    if (state.isRecording) {
                        setTimeout(() => {
                            try {
                                state.recognition.start();
                            } catch (e) {
                                console.error('Failed to restart speech recognition:', e);
                            }
                        }, 100);
                    }
                };

                try {
                    state.recognition.start();
                } catch (e) {
                    console.error('Failed to start speech recognition:', e);
                    updateStatusMessage('Speech recognition failed to start');
                }
            } else {
                updateStatusMessage('Speech recognition not supported');
            }

            // Start recording
            state.mediaRecorder.start(1000);
            state.isRecording = true;
            state.startTime = Date.now();
            state.keystrokes = [];
            state.transcription = [];
            state.gazeData = [];
            state.gameId = generateGameId();
            state.actionCounter = 0;

            // Resume WebGazer if available
            setWebGazerActive(true);

            updateRecordingUI();
        } catch (error) {
            console.error('Error starting recording:', error);
            alert('Could not access microphone. Recording will only capture keystrokes.');

            // Start keystroke-only recording
            state.isRecording = true;
            state.startTime = Date.now();
            state.keystrokes = [];
            state.transcription = [];
            state.gazeData = [];
            state.gameId = generateGameId();
            state.actionCounter = 0;

            // Resume WebGazer if available
            setWebGazerActive(true);

            updateRecordingUI();
        }
    }

    // Stop recording
    function stopRecording() {
        state.isRecording = false;

        if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') {
            state.mediaRecorder.stop();
            state.mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }

        if (state.recognition) {
            state.recognition.stop();
        }

        // Pause WebGazer
        setWebGazerActive(false);

        updateRecordingUI();
    }

    // Toggle recording
    function toggleRecording() {
        if (state.isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    }

    // Update recording UI
    function updateRecordingUI() {
        if (!elements.recordBtn) return;

        if (state.isRecording) {
            elements.recordBtn.textContent = 'Stop Recording';
            elements.recordBtn.classList.add('recording');
            elements.statusDiv.textContent = 'Recording gameplay and audio...';
            elements.statusDiv.classList.add('active');
        } else {
            elements.recordBtn.textContent = 'Start Recording';
            elements.recordBtn.classList.remove('recording');

            if (state.keystrokes.length > 0 || state.gazeData.length > 0) {
                const transcriptCount = state.transcription.length;
                const gazeCount = state.gazeData.length;
                elements.statusDiv.textContent = `Recording complete. ${state.keystrokes.length} actions, ${transcriptCount} speech, ${gazeCount} gaze points.`;
            } else {
                elements.statusDiv.textContent = 'Ready to record gameplay and audio';
            }
            elements.statusDiv.classList.remove('active');
        }
    }

    // Update status message
    function updateStatusMessage(message) {
        if (elements.statusDiv && state.isRecording) {
            elements.statusDiv.textContent = message;
        }
    }

    // Export recording
    function exportRecording() {
        if (state.keystrokes.length === 0 && state.gazeData.length === 0) {
            alert('No recording data to export');
            return;
        }

        const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');

        // Create export object matching the track-puzzle-session format
        // while also including all original data (game_id, reasoning, guid, etc.)
        const exportData = {
            sessionStart: new Date(state.startTime).toISOString(),
            sessionEnd: new Date().toISOString(),
            gameId: state.gameId,
            duration: Date.now() - state.startTime,
            events: state.keystrokes.map((ks, idx) => ({
                timestamp: new Date(state.startTime + ks.timestamp).toISOString(),
                level: ks.gameState.level || 1,
                key: ks.key,
                action: ks.action,
                gameStateBefore: ks.gameState.gameStateMatrix || null,
                selectedChunkPosition: ks.gameState.selectedChunkPosition || null,
                vehiclePos: ks.gameState.vehiclePos || null,
                goalPos: ks.gameState.goalPos || null,
                // Include original fields from gameFrames
                actionId: idx,
                reasoning: findNearbyReasoning(ks.timestamp),
                guid: generateGuid()
            })),
            transcription: state.transcription.map(t => ({
                timestamp: new Date(state.startTime + t.timestamp).toISOString(),
                text: t.text,
                confidence: t.confidence
            })),
            gaze: state.gazeData.map(g => ({
                x: g[0],
                y: g[1],
                timestamp: new Date(state.startTime + g[2]).toISOString()
            }))
        };

        // Custom JSON stringify that keeps grid rows on single lines
        function formatExportData(data) {
            const lines = [];
            lines.push('{');
            lines.push(`  "sessionStart": ${JSON.stringify(data.sessionStart)},`);
            lines.push(`  "sessionEnd": ${JSON.stringify(data.sessionEnd)},`);
            lines.push(`  "gameId": ${JSON.stringify(data.gameId)},`);
            lines.push(`  "duration": ${data.duration},`);

            // Events array
            lines.push('  "events": [');
            data.events.forEach((event, i) => {
                lines.push('    {');
                lines.push(`      "timestamp": ${JSON.stringify(event.timestamp)},`);
                lines.push(`      "level": ${event.level},`);
                lines.push(`      "key": ${JSON.stringify(event.key)},`);
                lines.push(`      "action": ${JSON.stringify(event.action)},`);

                // Format gameStateBefore with each row on one line
                if (event.gameStateBefore && Array.isArray(event.gameStateBefore)) {
                    lines.push('      "gameStateBefore": [');
                    event.gameStateBefore.forEach((row, rowIdx) => {
                        const comma = rowIdx < event.gameStateBefore.length - 1 ? ',' : '';
                        lines.push(`        ${JSON.stringify(row)}${comma}`);
                    });
                    lines.push('      ],');
                } else {
                    lines.push(`      "gameStateBefore": null,`);
                }

                lines.push(`      "selectedChunkPosition": ${JSON.stringify(event.selectedChunkPosition)},`);
                lines.push(`      "vehiclePos": ${JSON.stringify(event.vehiclePos)},`);
                lines.push(`      "goalPos": ${JSON.stringify(event.goalPos)},`);
                lines.push(`      "actionId": ${event.actionId},`);
                lines.push(`      "reasoning": ${JSON.stringify(event.reasoning)},`);
                lines.push(`      "guid": ${JSON.stringify(event.guid)}`);

                const eventComma = i < data.events.length - 1 ? ',' : '';
                lines.push(`    }${eventComma}`);
            });
            lines.push('  ],');

            // Transcription array
            lines.push('  "transcription": [');
            data.transcription.forEach((t, i) => {
                const comma = i < data.transcription.length - 1 ? ',' : '';
                lines.push(`    {`);
                lines.push(`      "timestamp": ${JSON.stringify(t.timestamp)},`);
                lines.push(`      "text": ${JSON.stringify(t.text)},`);
                lines.push(`      "confidence": ${t.confidence}`);
                lines.push(`    }${comma}`);
            });
            lines.push('  ],');

            // Gaze array
            lines.push('  "gaze": [');
            data.gaze.forEach((g, i) => {
                const comma = i < data.gaze.length - 1 ? ',' : '';
                lines.push(`    {`);
                lines.push(`      "x": ${g.x !== null ? g.x : 'null'},`);
                lines.push(`      "y": ${g.y !== null ? g.y : 'null'},`);
                lines.push(`      "timestamp": ${JSON.stringify(g.timestamp)}`);
                lines.push(`    }${comma}`);
            });
            lines.push('  ]');

            lines.push('}');
            return lines.join('\n');
        }

        const jsonString = formatExportData(exportData);
        const jsonBlob = new Blob([jsonString], { type: 'application/json' });
        const jsonUrl = URL.createObjectURL(jsonBlob);
        const jsonLink = document.createElement('a');
        jsonLink.href = jsonUrl;
        jsonLink.download = `${config.gamePrefix}-session-${timestamp}.json`;
        document.body.appendChild(jsonLink);
        jsonLink.click();
        document.body.removeChild(jsonLink);
        URL.revokeObjectURL(jsonUrl);

        // Download audio if available
        if (state.audioChunks.length > 0) {
            const audioBlob = new Blob(state.audioChunks, { type: 'audio/webm' });
            const audioUrl = URL.createObjectURL(audioBlob);
            const audioLink = document.createElement('a');
            audioLink.href = audioUrl;
            audioLink.download = `${config.gamePrefix}-audio-${timestamp}.webm`;
            document.body.appendChild(audioLink);
            audioLink.click();
            document.body.removeChild(audioLink);
            URL.revokeObjectURL(audioUrl);
        }

        const audioMsg = state.audioChunks.length > 0 ? '\n- WebM: audio recording' : '';
        const gazeMsg = state.gazeData.length > 0 ? `\n- ${state.gazeData.length} gaze points` : '';
        const transcriptMsg = state.transcription.length > 0 ? `\n- ${state.transcription.length} transcriptions` : '';
        alert(`Exported ${state.keystrokes.length} events!${transcriptMsg}${gazeMsg}\n\nFiles:\n- JSON: session data${audioMsg}`);
    }

    // Create and inject the recording UI
    function createUI() {
        // Add CSS styles
        const style = document.createElement('style');
        style.textContent = `
            .audio-recorder-controls {
                margin: 15px 0;
                text-align: center;
            }
            .audio-recorder-btn {
                background-color: #333;
                border: 2px solid #555;
                color: white;
                padding: 10px 20px;
                margin: 0 5px;
                border-radius: 5px;
                cursor: pointer;
                font-family: inherit;
                font-size: 14px;
            }
            .audio-recorder-btn:hover {
                background-color: #555;
            }
            .audio-recorder-btn.recording {
                background-color: #ff0000;
                border-color: #ff3333;
                animation: audio-recorder-pulse 1.5s infinite;
            }
            .audio-recorder-btn.disabled {
                background-color: #666;
                border-color: #888;
                cursor: not-allowed;
                opacity: 0.5;
            }
            @keyframes audio-recorder-pulse {
                0% { opacity: 1; }
                50% { opacity: 0.7; }
                100% { opacity: 1; }
            }
            .audio-recorder-status {
                margin-top: 10px;
                font-size: 12px;
                color: #666;
            }
            .audio-recorder-status.active {
                color: #ff6666;
            }
        `;
        document.head.appendChild(style);

        // Create container
        elements.container = document.createElement('div');
        elements.container.className = 'audio-recorder-controls';

        // Create record button
        elements.recordBtn = document.createElement('button');
        elements.recordBtn.className = 'audio-recorder-btn';
        elements.recordBtn.textContent = 'Start Recording';
        elements.recordBtn.onclick = toggleRecording;

        // Create status div
        elements.statusDiv = document.createElement('div');
        elements.statusDiv.className = 'audio-recorder-status';
        elements.statusDiv.textContent = 'Ready to record gameplay and audio';

        // Assemble container
        elements.container.appendChild(elements.recordBtn);
        elements.container.appendChild(elements.statusDiv);
    }

    // Setup keyboard event listeners
    function setupKeyboardListeners() {
        document.addEventListener('keydown', (e) => {
            if (state.isRecording) {
                // Only log keys that have a mapped action
                const action = config.keyActionMap[e.key];
                if (action) {
                    const timestamp = Date.now() - state.startTime;
                    recordKeystroke(e.key, action, timestamp);
                }
            }
        });

        // No keyup logging - we only care about keydown actions
    }

    // Initialize the recorder
    async function init(userConfig = {}) {
        // Merge user config
        config = { ...config, ...userConfig };

        // Create UI
        createUI();

        // Setup keyboard listeners
        setupKeyboardListeners();

        // Initialize WebGazer (async, non-blocking)
        initWebGazer().then(success => {
            if (success) {
                updateStatusMessage('Ready to record (eye tracking enabled)');
                // Pause until recording starts
                setWebGazerActive(false);
            }
        });

        // Insert UI into page
        let inserted = false;

        // If insertBeforeSelector is specified, try to insert before that element
        if (config.insertBeforeSelector) {
            const target = document.querySelector(config.insertBeforeSelector);
            if (target && target.parentNode) {
                target.parentNode.insertBefore(elements.container, target);
                inserted = true;
            }
        }

        // Fall back to common insertion points
        if (!inserted) {
            const insertionPoints = [
                '#game-container',
                '#instructions',
                '.game-container',
                'body'
            ];

            for (const selector of insertionPoints) {
                const target = document.querySelector(selector);
                if (target) {
                    if (selector === 'body') {
                        // Insert after first child for body
                        if (target.firstChild) {
                            target.insertBefore(elements.container, target.firstChild.nextSibling);
                        } else {
                            target.appendChild(elements.container);
                        }
                    } else {
                        // Insert after the found element
                        target.parentNode.insertBefore(elements.container, target.nextSibling);
                    }
                    inserted = true;
                    break;
                }
            }
        }

        if (!inserted) {
            document.body.appendChild(elements.container);
        }

        return {
            recordKeystroke,
            startRecording,
            stopRecording,
            exportRecording,
            isRecording: () => state.isRecording,
            getContainer: () => elements.container
        };
    }

    // Public API
    return {
        init,
        recordKeystroke: (key, action) => {
            if (state.isRecording) {
                const timestamp = Date.now() - state.startTime;
                recordKeystroke(key, action, timestamp);
            }
        },
        isRecording: () => state.isRecording,
        getState: () => ({ ...state }),
        exportRecording
    };
})();
