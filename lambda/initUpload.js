/**
 * Lambda function: initUpload
 *
 * Initializes an upload session by generating presigned URLs for S3.
 *
 * Request body: { sessionCode: string, gameName: string }
 * Response: { sessionId, basePath, audioUploadUrl, audioKey }
 */

const { S3Client, PutObjectCommand } = require('@aws-sdk/client-s3');
const { getSignedUrl } = require('@aws-sdk/s3-request-presigner');
const { getCorsHeaders } = require('./cors');

const s3Client = new S3Client({ region: process.env.AWS_REGION || 'us-east-1' });
const BUCKET_NAME = process.env.BUCKET_NAME || 'gamestudy-data';
const EXPIRATION_SECONDS = 10800; // 3 hours

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
        const { sessionCode, gameName } = body;

        if (!sessionCode || !gameName) {
            return {
                statusCode: 400,
                headers: corsHeaders,
                body: JSON.stringify({ error: 'sessionCode and gameName are required' })
            };
        }

        // Generate paths
        const date = new Date().toISOString().split('T')[0];
        const sessionId = `${sessionCode}-${gameName}`;
        const basePath = `sessions/${date}/${sessionId}`;
        const audioKey = `${basePath}/audio.webm`;

        // Generate presigned URL for audio upload
        const putCommand = new PutObjectCommand({
            Bucket: BUCKET_NAME,
            Key: audioKey,
            ContentType: 'audio/webm'
        });

        const audioUploadUrl = await getSignedUrl(s3Client, putCommand, {
            expiresIn: EXPIRATION_SECONDS
        });

        return {
            statusCode: 200,
            headers: corsHeaders,
            body: JSON.stringify({
                sessionId,
                basePath,
                audioUploadUrl,
                audioKey
            })
        };

    } catch (error) {
        console.error('initUpload error:', error);
        return {
            statusCode: 500,
            headers: corsHeaders,
            body: JSON.stringify({ error: error.message })
        };
    }
};
