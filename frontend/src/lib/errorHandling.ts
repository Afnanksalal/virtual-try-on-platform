/**
 * Comprehensive error handling utilities for Studio UX
 * Provides retry logic, error mapping, and user-friendly messages
 */

import { toast } from 'sonner';
import { APIError } from './api';

// Error types
export enum ErrorType {
  NETWORK = 'NETWORK',
  VALIDATION = 'VALIDATION',
  AUTHENTICATION = 'AUTHENTICATION',
  STORAGE = 'STORAGE',
  ML_PROCESSING = 'ML_PROCESSING',
  UNKNOWN = 'UNKNOWN',
}

// Error severity levels
export enum ErrorSeverity {
  INFO = 'INFO',
  WARNING = 'WARNING',
  ERROR = 'ERROR',
  CRITICAL = 'CRITICAL',
}

// Error context for logging
export interface ErrorContext {
  operation: string;
  userId?: string;
  timestamp: Date;
  errorType: ErrorType;
  severity: ErrorSeverity;
  originalError: unknown;
  additionalData?: Record<string, unknown>;
}

// User-friendly error messages mapping
const ERROR_MESSAGES: Record<string, string> = {
  // Network errors
  'Request timeout': 'The request took too long. Please check your connection and try again.',
  'Failed to fetch': 'Unable to connect to the server. Please check your internet connection.',
  'Network request failed': 'Network error occurred. Please check your connection and try again.',
  
  // Authentication errors
  'Unauthorized': 'Your session has expired. Please log in again.',
  'Forbidden': 'You don\'t have permission to perform this action.',
  'Invalid token': 'Your session is invalid. Please log in again.',
  
  // Validation errors
  'File too large': 'File size must be under 10MB. Please choose a smaller file.',
  'Invalid file type': 'Please upload a valid image file (JPG or PNG).',
  'Missing required field': 'Please fill in all required fields.',
  'Invalid format': 'The file format is not supported. Please use JPG or PNG.',
  
  // Storage errors
  'Upload failed': 'Failed to upload the file. Please try again.',
  'Delete failed': 'Failed to delete the item. Please try again.',
  'Storage quota exceeded': 'You\'ve reached your storage limit. Please delete some items or upgrade.',
  
  // ML Processing errors
  'Processing failed': 'Unable to process the image. Please try with a different image.',
  'Model not available': 'The AI model is temporarily unavailable. Please try again later.',
  'Invalid image': 'The image could not be processed. Please use a clear, well-lit photo.',
  
  // Generic errors
  'Unknown error': 'Something went wrong. Please try again.',
  'Server error': 'A server error occurred. Our team has been notified.',
};

/**
 * Get user-friendly error message from error object
 */
export function getUserFriendlyMessage(error: unknown): string {
  if (error instanceof APIError) {
    // Check for specific error messages in the API error
    const errorMessage = error.message.toLowerCase();
    
    for (const [key, message] of Object.entries(ERROR_MESSAGES)) {
      if (errorMessage.includes(key.toLowerCase())) {
        return message;
      }
    }
    
    // Status code based messages
    if (error.status === 401 || error.status === 403) {
      return ERROR_MESSAGES['Unauthorized'];
    }
    if (error.status === 408) {
      return ERROR_MESSAGES['Request timeout'];
    }
    if (error.status >= 500) {
      return ERROR_MESSAGES['Server error'];
    }
  }
  
  if (error instanceof Error) {
    // Check error message against known patterns
    const errorMessage = error.message.toLowerCase();
    
    for (const [key, message] of Object.entries(ERROR_MESSAGES)) {
      if (errorMessage.includes(key.toLowerCase())) {
        return message;
      }
    }
  }
  
  return ERROR_MESSAGES['Unknown error'];
}

/**
 * Classify error type
 */
export function classifyError(error: unknown): ErrorType {
  if (error instanceof APIError) {
    if (error.status === 401 || error.status === 403) {
      return ErrorType.AUTHENTICATION;
    }
    if (error.status === 408 || error.status === 0) {
      return ErrorType.NETWORK;
    }
    if (error.status === 400 || error.status === 422) {
      return ErrorType.VALIDATION;
    }
    if (error.status >= 500) {
      return ErrorType.ML_PROCESSING;
    }
  }
  
  if (error instanceof Error) {
    const message = error.message.toLowerCase();
    if (message.includes('network') || message.includes('fetch') || message.includes('timeout')) {
      return ErrorType.NETWORK;
    }
    if (message.includes('validation') || message.includes('invalid')) {
      return ErrorType.VALIDATION;
    }
    if (message.includes('auth') || message.includes('unauthorized')) {
      return ErrorType.AUTHENTICATION;
    }
    if (message.includes('storage') || message.includes('upload') || message.includes('delete')) {
      return ErrorType.STORAGE;
    }
  }
  
  return ErrorType.UNKNOWN;
}

/**
 * Determine error severity
 */
export function getErrorSeverity(errorType: ErrorType): ErrorSeverity {
  switch (errorType) {
    case ErrorType.AUTHENTICATION:
      return ErrorSeverity.CRITICAL;
    case ErrorType.NETWORK:
    case ErrorType.STORAGE:
      return ErrorSeverity.WARNING;
    case ErrorType.VALIDATION:
      return ErrorSeverity.INFO;
    case ErrorType.ML_PROCESSING:
      return ErrorSeverity.ERROR;
    default:
      return ErrorSeverity.ERROR;
  }
}

/**
 * Log error with context
 */
export function logError(context: ErrorContext): void {
  const logData = {
    ...context,
    timestamp: context.timestamp.toISOString(),
    userAgent: typeof window !== 'undefined' ? window.navigator.userAgent : 'unknown',
  };
  
  // Console logging with appropriate level
  switch (context.severity) {
    case ErrorSeverity.CRITICAL:
      console.error('[CRITICAL ERROR]', logData);
      break;
    case ErrorSeverity.ERROR:
      console.error('[ERROR]', logData);
      break;
    case ErrorSeverity.WARNING:
      console.warn('[WARNING]', logData);
      break;
    case ErrorSeverity.INFO:
      console.info('[INFO]', logData);
      break;
  }
  
  // In production, you would send this to a logging service
  // Example: sendToLoggingService(logData);
}

/**
 * Handle error with toast notification and logging
 */
export function handleError(
  error: unknown,
  operation: string,
  options: {
    userId?: string;
    showToast?: boolean;
    additionalData?: Record<string, unknown>;
  } = {}
): void {
  const { userId, showToast = true, additionalData } = options;
  
  const errorType = classifyError(error);
  const severity = getErrorSeverity(errorType);
  const userMessage = getUserFriendlyMessage(error);
  
  // Log error
  logError({
    operation,
    userId,
    timestamp: new Date(),
    errorType,
    severity,
    originalError: error,
    additionalData,
  });
  
  // Show toast notification
  if (showToast) {
    toast.error(userMessage);
  }
}

/**
 * Retry configuration
 */
export interface RetryConfig {
  maxAttempts: number;
  initialDelay: number;
  maxDelay: number;
  backoffMultiplier: number;
  retryableErrors?: ErrorType[];
}

const DEFAULT_RETRY_CONFIG: RetryConfig = {
  maxAttempts: 3,
  initialDelay: 1000, // 1 second
  maxDelay: 10000, // 10 seconds
  backoffMultiplier: 2,
  retryableErrors: [ErrorType.NETWORK, ErrorType.UNKNOWN],
};

/**
 * Check if error is retryable
 */
function isRetryableError(error: unknown, retryableErrors: ErrorType[]): boolean {
  const errorType = classifyError(error);
  return retryableErrors.includes(errorType);
}

/**
 * Calculate delay for next retry with exponential backoff
 */
function calculateDelay(attempt: number, config: RetryConfig): number {
  const delay = config.initialDelay * Math.pow(config.backoffMultiplier, attempt - 1);
  return Math.min(delay, config.maxDelay);
}

/**
 * Sleep utility
 */
function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Retry wrapper with exponential backoff
 */
export async function withRetry<T>(
  operation: () => Promise<T>,
  operationName: string,
  config: Partial<RetryConfig> = {}
): Promise<T> {
  const finalConfig: RetryConfig = { ...DEFAULT_RETRY_CONFIG, ...config };
  let lastError: unknown;
  
  for (let attempt = 1; attempt <= finalConfig.maxAttempts; attempt++) {
    try {
      return await operation();
    } catch (error) {
      lastError = error;
      
      // Check if we should retry
      const shouldRetry = 
        attempt < finalConfig.maxAttempts &&
        isRetryableError(error, finalConfig.retryableErrors || DEFAULT_RETRY_CONFIG.retryableErrors!);
      
      if (!shouldRetry) {
        throw error;
      }
      
      // Calculate delay and wait
      const delay = calculateDelay(attempt, finalConfig);
      
      console.warn(
        `[RETRY] Attempt ${attempt}/${finalConfig.maxAttempts} failed for ${operationName}. ` +
        `Retrying in ${delay}ms...`,
        error
      );
      
      // Show toast for retry
      if (attempt === 1) {
        toast.info(`Retrying ${operationName}...`);
      }
      
      await sleep(delay);
    }
  }
  
  // All attempts failed
  throw lastError;
}

/**
 * Success notification helper
 */
export function showSuccess(message: string): void {
  toast.success(message);
}

/**
 * Info notification helper
 */
export function showInfo(message: string): void {
  toast.info(message);
}

/**
 * Warning notification helper
 */
export function showWarning(message: string): void {
  toast.warning(message);
}
