/**
 * Lambda function: submitSession
 *
 * Receives session data and eye tracking data, writes them to S3.
 *
 * Request body: { basePath: string, sessionData: object, eyeTrackingData: object, audioKey: string }
 * Response: { success: boolean, paths: object }
 */

const { S3Client, PutObjectCommand } = require('@aws-sdk/client-s3');
const { getCorsHeaders } = require('./cors');

const s3Client = new S3Client({ region: process.env.AWS_REGION || 'us-east-1' });
const BUCKET_NAME = process.env.BUCKET_NAME || 'gamestudy-data';

exports.handler = async (event) => {
    const corsHeaders = getCorsHeaders(event, 'POST, OPTIONS');

    // Handle CORS preflight
    if (event.httpMethod === 'OPTIONS') {
        return {
            statusCode: 200,
            headers: corsHeaders,
            body: ''
        };
    }

    try {
        const body = JSON.parse(event.body);
        const { basePath, sessionData, eyeTrackingData, audioKey } = body;

        if (!basePath) {
            return {
                statusCode: 400,
                headers: corsHeaders,
                body: JSON.stringify({ error: 'basePath is required' })
            };
        }

        const uploads = [];
        const paths = {};

        // Upload session data
        if (sessionData) {
            const sessionKey = `${basePath}/session-data.json`;
            uploads.push(
                s3Client.send(new PutObjectCommand({
                    Bucket: BUCKET_NAME,
                    Key: sessionKey,
                    Body: JSON.stringify(sessionData, null, 2),
                    ContentType: 'application/json'
                }))
            );
            paths.sessionData = sessionKey;
        }

        // Upload eye tracking data
        if (eyeTrackingData) {
            const eyeTrackingKey = `${basePath}/eye-tracking.json`;
            uploads.push(
                s3Client.send(new PutObjectCommand({
                    Bucket: BUCKET_NAME,
                    Key: eyeTrackingKey,
                    Body: JSON.stringify(eyeTrackingData, null, 2),
                    ContentType: 'application/json'
                }))
            );
            paths.eyeTracking = eyeTrackingKey;
        }

        // Record audio key if provided
        if (audioKey) {
            paths.audio = audioKey;
        }

        // Execute all uploads
        await Promise.all(uploads);

        return {
            statusCode: 200,
            headers: corsHeaders,
            body: JSON.stringify({
                success: true,
                message: 'Session data uploaded successfully',
                paths
            })
        };

    } catch (error) {
        console.error('submitSession error:', error);

        // Attempt to store failed data for recovery
        try {
            const body = JSON.parse(event.body);
            const failedKey = `failed/${Date.now()}-${body.basePath?.split('/').pop() || 'unknown'}/data.json`;
            await s3Client.send(new PutObjectCommand({
                Bucket: BUCKET_NAME,
                Key: failedKey,
                Body: JSON.stringify({
                    sessionData: body.sessionData,
                    eyeTrackingData: body.eyeTrackingData,
                    error: error.message,
                    timestamp: new Date().toISOString()
                }),
                ContentType: 'application/json'
            }));
        } catch (recoveryError) {
            console.error('Failed to store recovery data:', recoveryError);
        }

        return {
            statusCode: 500,
            headers: corsHeaders,
            body: JSON.stringify({ success: false, error: error.message })
        };
    }
};
