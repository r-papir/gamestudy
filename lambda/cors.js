/**
 * Shared CORS configuration for all Lambda functions.
 *
 * Single source of truth for allowed origins — also consumed by sync-cors.sh
 * to keep the S3 bucket CORS policy in sync.
 */

const ALLOWED_ORIGINS = [
    'https://marksantolucito.com',
    'https://www.marksantolucito.com',
    'http://marksantolucito.com',
    'http://www.marksantolucito.com',
    'https://r-papir.github.io',
    'http://r-papir.github.io'
];

/**
 * Build CORS response headers for the given request.
 *
 * @param {object} event  – API Gateway event
 * @param {string} methods – Allowed HTTP methods (e.g. 'POST, OPTIONS')
 * @returns {object} headers object
 */
function getCorsHeaders(event, methods) {
    const origin = event.headers?.origin || event.headers?.Origin || '';
    return {
        'Access-Control-Allow-Origin': ALLOWED_ORIGINS.includes(origin) ? origin : ALLOWED_ORIGINS[0],
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': methods
    };
}

module.exports = { ALLOWED_ORIGINS, getCorsHeaders };
