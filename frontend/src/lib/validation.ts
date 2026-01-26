/**
 * Input validation utilities for Studio UX
 * Provides validation functions with user-friendly error messages
 */

export interface ValidationResult {
  isValid: boolean;
  error?: string;
}

export interface FileValidationOptions {
  maxSizeMB?: number;
  allowedTypes?: string[];
  minWidth?: number;
  minHeight?: number;
  maxWidth?: number;
  maxHeight?: number;
}

const DEFAULT_FILE_OPTIONS: FileValidationOptions = {
  maxSizeMB: 10,
  allowedTypes: ['image/jpeg', 'image/jpg', 'image/png'],
};

/**
 * Validate file type
 */
export function validateFileType(file: File, allowedTypes: string[]): ValidationResult {
  if (!allowedTypes.includes(file.type)) {
    const typeNames = allowedTypes
      .map(type => type.split('/')[1].toUpperCase())
      .join(', ');
    return {
      isValid: false,
      error: `Please upload a valid image file (${typeNames})`,
    };
  }
  
  return { isValid: true };
}

/**
 * Validate file size
 */
export function validateFileSize(file: File, maxSizeMB: number): ValidationResult {
  const maxSizeBytes = maxSizeMB * 1024 * 1024;
  
  if (file.size > maxSizeBytes) {
    return {
      isValid: false,
      error: `File size must be under ${maxSizeMB}MB. Your file is ${(file.size / 1024 / 1024).toFixed(1)}MB`,
    };
  }
  
  return { isValid: true };
}

/**
 * Validate image dimensions
 */
export async function validateImageDimensions(
  file: File,
  options: Pick<FileValidationOptions, 'minWidth' | 'minHeight' | 'maxWidth' | 'maxHeight'>
): Promise<ValidationResult> {
  return new Promise((resolve) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    
    img.onload = () => {
      URL.revokeObjectURL(url);
      
      const { minWidth, minHeight, maxWidth, maxHeight } = options;
      
      if (minWidth && img.width < minWidth) {
        resolve({
          isValid: false,
          error: `Image width must be at least ${minWidth}px. Your image is ${img.width}px wide`,
        });
        return;
      }
      
      if (minHeight && img.height < minHeight) {
        resolve({
          isValid: false,
          error: `Image height must be at least ${minHeight}px. Your image is ${img.height}px tall`,
        });
        return;
      }
      
      if (maxWidth && img.width > maxWidth) {
        resolve({
          isValid: false,
          error: `Image width must be at most ${maxWidth}px. Your image is ${img.width}px wide`,
        });
        return;
      }
      
      if (maxHeight && img.height > maxHeight) {
        resolve({
          isValid: false,
          error: `Image height must be at most ${maxHeight}px. Your image is ${img.height}px tall`,
        });
        return;
      }
      
      resolve({ isValid: true });
    };
    
    img.onerror = () => {
      URL.revokeObjectURL(url);
      resolve({
        isValid: false,
        error: 'Unable to read image file. Please try a different file',
      });
    };
    
    img.src = url;
  });
}

/**
 * Comprehensive file validation
 */
export async function validateFile(
  file: File,
  options: FileValidationOptions = {}
): Promise<ValidationResult> {
  const finalOptions = { ...DEFAULT_FILE_OPTIONS, ...options };
  
  // Check if file exists
  if (!file) {
    return {
      isValid: false,
      error: 'Please select a file',
    };
  }
  
  // Validate file type
  if (finalOptions.allowedTypes) {
    const typeResult = validateFileType(file, finalOptions.allowedTypes);
    if (!typeResult.isValid) {
      return typeResult;
    }
  }
  
  // Validate file size
  if (finalOptions.maxSizeMB) {
    const sizeResult = validateFileSize(file, finalOptions.maxSizeMB);
    if (!sizeResult.isValid) {
      return sizeResult;
    }
  }
  
  // Validate dimensions if specified
  if (
    finalOptions.minWidth ||
    finalOptions.minHeight ||
    finalOptions.maxWidth ||
    finalOptions.maxHeight
  ) {
    const dimensionsResult = await validateImageDimensions(file, finalOptions);
    if (!dimensionsResult.isValid) {
      return dimensionsResult;
    }
  }
  
  return { isValid: true };
}

/**
 * Validate email format
 */
export function validateEmail(email: string): ValidationResult {
  if (!email) {
    return {
      isValid: false,
      error: 'Email is required',
    };
  }
  
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  
  if (!emailRegex.test(email)) {
    return {
      isValid: false,
      error: 'Please enter a valid email address',
    };
  }
  
  return { isValid: true };
}

/**
 * Validate required field
 */
export function validateRequired(value: unknown, fieldName: string): ValidationResult {
  if (value === null || value === undefined || value === '') {
    return {
      isValid: false,
      error: `${fieldName} is required`,
    };
  }
  
  return { isValid: true };
}

/**
 * Validate number range
 */
export function validateNumberRange(
  value: number,
  min: number,
  max: number,
  fieldName: string
): ValidationResult {
  if (isNaN(value)) {
    return {
      isValid: false,
      error: `${fieldName} must be a valid number`,
    };
  }
  
  if (value < min || value > max) {
    return {
      isValid: false,
      error: `${fieldName} must be between ${min} and ${max}`,
    };
  }
  
  return { isValid: true };
}

/**
 * Validate string length
 */
export function validateStringLength(
  value: string,
  minLength: number,
  maxLength: number,
  fieldName: string
): ValidationResult {
  if (value.length < minLength) {
    return {
      isValid: false,
      error: `${fieldName} must be at least ${minLength} characters`,
    };
  }
  
  if (value.length > maxLength) {
    return {
      isValid: false,
      error: `${fieldName} must be at most ${maxLength} characters`,
    };
  }
  
  return { isValid: true };
}

/**
 * Validate URL format
 */
export function validateURL(url: string): ValidationResult {
  if (!url) {
    return {
      isValid: false,
      error: 'URL is required',
    };
  }
  
  try {
    new URL(url);
    return { isValid: true };
  } catch {
    return {
      isValid: false,
      error: 'Please enter a valid URL',
    };
  }
}

/**
 * Batch validation helper
 */
export interface ValidationRule {
  field: string;
  value: unknown;
  validator: (value: unknown) => ValidationResult | Promise<ValidationResult>;
}

export async function validateBatch(rules: ValidationRule[]): Promise<{
  isValid: boolean;
  errors: Record<string, string>;
}> {
  const errors: Record<string, string> = {};
  
  for (const rule of rules) {
    const result = await rule.validator(rule.value);
    if (!result.isValid && result.error) {
      errors[rule.field] = result.error;
    }
  }
  
  return {
    isValid: Object.keys(errors).length === 0,
    errors,
  };
}

/**
 * Garment image validation (specific to our use case)
 */
export async function validateGarmentImage(file: File): Promise<ValidationResult> {
  return validateFile(file, {
    maxSizeMB: 10,
    allowedTypes: ['image/jpeg', 'image/jpg', 'image/png'],
  });
}

/**
 * Personal image validation (specific to our use case)
 */
export async function validatePersonalImage(file: File): Promise<ValidationResult> {
  return validateFile(file, {
    maxSizeMB: 10,
    allowedTypes: ['image/jpeg', 'image/jpg', 'image/png'],
  });
}

/**
 * Body parameters validation
 */
export interface BodyParametersValidation {
  ethnicity: string;
  bodyType: string;
  height: number;
  weight: number;
}

export function validateBodyParameters(params: BodyParametersValidation): {
  isValid: boolean;
  errors: Record<string, string>;
} {
  const errors: Record<string, string> = {};
  
  // Validate ethnicity
  const ethnicityResult = validateRequired(params.ethnicity, 'Ethnicity');
  if (!ethnicityResult.isValid && ethnicityResult.error) {
    errors.ethnicity = ethnicityResult.error;
  }
  
  // Validate body type
  const bodyTypeResult = validateRequired(params.bodyType, 'Body type');
  if (!bodyTypeResult.isValid && bodyTypeResult.error) {
    errors.bodyType = bodyTypeResult.error;
  }
  
  // Validate height (100-250 cm)
  const heightResult = validateNumberRange(params.height, 100, 250, 'Height');
  if (!heightResult.isValid && heightResult.error) {
    errors.height = heightResult.error;
  }
  
  // Validate weight (30-200 kg)
  const weightResult = validateNumberRange(params.weight, 30, 200, 'Weight');
  if (!weightResult.isValid && weightResult.error) {
    errors.weight = weightResult.error;
  }
  
  return {
    isValid: Object.keys(errors).length === 0,
    errors,
  };
}
