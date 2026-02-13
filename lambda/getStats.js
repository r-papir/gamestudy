/**
 * Lambda function: getStats
 *
 * Lists objects in the S3 bucket and returns aggregated session stats,
 * including per-session detail metrics (audio length, action count, gaze points).
 *
 * Response: { totalSessions, totalFiles, sessionsByGame, sessionsByDate,
 *             mostRecentUpload, averages, sessionDetails }
 */

const { S3Client, ListObjectsV2Command, GetObjectCommand } = require('@aws-sdk/client-s3');
const { getCorsHeaders } = require('./cors');

const s3Client = new S3Client({ region: process.env.AWS_REGION || 'us-east-1' });
const BUCKET_NAME = process.env.BUCKET_NAME || 'gamestudy-data';

/**
 * Read and parse a JSON object from S3. Returns null on any error.
 */
async function readJsonFromS3(key) {
    try {
        const response = await s3Client.send(new GetObjectCommand({
            Bucket: BUCKET_NAME,
            Key: key
        }));
        const body = await response.Body.transformToString();
        return JSON.parse(body);
    } catch {
        return null;
    }
}

exports.handler = async (event) => {
    const corsHeaders = getCorsHeaders(event, 'GET, OPTIONS');

    // Handle CORS preflight
    if (event.httpMethod === 'OPTIONS') {
        return {
            statusCode: 200,
            headers: corsHeaders,
            body: ''
        };
    }

    try {
        // List all objects under sessions/
        const objects = [];
        let continuationToken;

        do {
            const command = new ListObjectsV2Command({
                Bucket: BUCKET_NAME,
                Prefix: 'sessions/',
                ContinuationToken: continuationToken
            });
            const response = await s3Client.send(command);
            if (response.Contents) {
                objects.push(...response.Contents);
            }
            continuationToken = response.IsTruncated ? response.NextContinuationToken : undefined;
        } while (continuationToken);

        // Group objects by session path: "sessions/{date}/{sessionId}"
        // Track per-session files and audio sizes
        const sessionMap = {}; // key: "date/sessionId" -> { date, sessionId, gameName, files: {}, audioSize }
        const sessionsByGame = {};
        const sessionsByDate = {};
        let mostRecentUpload = null;

        for (const obj of objects) {
            const parts = obj.Key.split('/');
            // parts: ["sessions", date, sessionId, filename]
            if (parts.length < 4) continue;

            const date = parts[1];
            const sessionId = parts[2];
            const filename = parts[3];
            const sessionKey = `${date}/${sessionId}`;

            // Extract game name: sessionId is {code}-{game}
            const dashIndex = sessionId.indexOf('-');
            const gameName = dashIndex !== -1 ? sessionId.substring(dashIndex + 1) : sessionId;

            if (!sessionMap[sessionKey]) {
                sessionMap[sessionKey] = { date, sessionId, gameName, files: {}, audioSize: 0 };
            }

            // Track which files exist for this session
            if (filename === 'session-data.json') {
                sessionMap[sessionKey].files.sessionData = obj.Key;
            } else if (filename === 'eye-tracking.json') {
                sessionMap[sessionKey].files.eyeTracking = obj.Key;
            } else if (filename === 'audio.webm') {
                sessionMap[sessionKey].audioSize = obj.Size || 0;
            }

            if (!sessionsByGame[gameName]) sessionsByGame[gameName] = new Set();
            sessionsByGame[gameName].add(sessionKey);

            if (!sessionsByDate[date]) sessionsByDate[date] = new Set();
            sessionsByDate[date].add(sessionKey);

            if (!mostRecentUpload || obj.LastModified > mostRecentUpload) {
                mostRecentUpload = obj.LastModified;
            }
        }

        // Read session detail files in parallel (bounded concurrency)
        const sessionKeys = Object.keys(sessionMap);
        const sessionDetails = [];

        // Process sessions in batches of 10 to avoid overwhelming S3
        const BATCH_SIZE = 10;
        for (let i = 0; i < sessionKeys.length; i += BATCH_SIZE) {
            const batch = sessionKeys.slice(i, i + BATCH_SIZE);
            const results = await Promise.all(batch.map(async (sk) => {
                const info = sessionMap[sk];
                const detail = {
                    sessionId: info.sessionId,
                    date: info.date,
                    game: info.gameName,
                    audioSizeBytes: info.audioSize,
                    durationMs: null,
                    actionCount: null,
                    gazePointCount: null
                };

                // Read session-data.json for action (movement) count
                if (info.files.sessionData) {
                    const sessionData = await readJsonFromS3(info.files.sessionData);
                    if (sessionData) {
                        detail.actionCount = Array.isArray(sessionData.movements)
                            ? sessionData.movements.length : 0;
                        // Fallback duration from session timestamps
                        if (sessionData.sessionStart && sessionData.sessionEnd) {
                            const start = new Date(sessionData.sessionStart).getTime();
                            const end = new Date(sessionData.sessionEnd).getTime();
                            if (!isNaN(start) && !isNaN(end)) {
                                detail.durationMs = end - start;
                            }
                        }
                    }
                }

                // Read eye-tracking.json for gaze count and duration
                if (info.files.eyeTracking) {
                    const etData = await readJsonFromS3(info.files.eyeTracking);
                    if (etData) {
                        detail.gazePointCount = Array.isArray(etData.gaze)
                            ? etData.gaze.length : 0;
                        // Prefer the explicit duration field from eye-tracking
                        if (typeof etData.duration === 'number') {
                            detail.durationMs = etData.duration;
                        }
                    }
                }

                return detail;
            }));
            sessionDetails.push(...results);
        }

        // Compute averages
        const withDuration = sessionDetails.filter(d => d.durationMs != null && d.durationMs > 0);
        const withActions = sessionDetails.filter(d => d.actionCount != null);
        const withGaze = sessionDetails.filter(d => d.gazePointCount != null);
        const withAudio = sessionDetails.filter(d => d.audioSizeBytes > 0);

        const averages = {
            durationMs: withDuration.length > 0
                ? Math.round(withDuration.reduce((s, d) => s + d.durationMs, 0) / withDuration.length)
                : null,
            actionCount: withActions.length > 0
                ? Math.round(withActions.reduce((s, d) => s + d.actionCount, 0) / withActions.length)
                : null,
            gazePointCount: withGaze.length > 0
                ? Math.round(withGaze.reduce((s, d) => s + d.gazePointCount, 0) / withGaze.length)
                : null,
            audioSizeBytes: withAudio.length > 0
                ? Math.round(withAudio.reduce((s, d) => s + d.audioSizeBytes, 0) / withAudio.length)
                : null
        };

        // Convert sets to counts
        const sessionsByGameCounts = {};
        for (const [game, set] of Object.entries(sessionsByGame)) {
            sessionsByGameCounts[game] = set.size;
        }
        const sessionsByDateCounts = {};
        for (const [date, set] of Object.entries(sessionsByDate)) {
            sessionsByDateCounts[date] = set.size;
        }

        return {
            statusCode: 200,
            headers: corsHeaders,
            body: JSON.stringify({
                totalSessions: sessionKeys.length,
                totalFiles: objects.length,
                sessionsByGame: sessionsByGameCounts,
                sessionsByDate: sessionsByDateCounts,
                mostRecentUpload: mostRecentUpload ? mostRecentUpload.toISOString() : null,
                averages,
                sessionDetails
            })
        };

    } catch (error) {
        console.error('getStats error:', error);
        return {
            statusCode: 500,
            headers: corsHeaders,
            body: JSON.stringify({ error: error.message })
        };
    }
};
