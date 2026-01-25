# Implementation Plan: Codebase Modernization

## Overview

This implementation plan transforms the virtual try-on platform into a production-ready system through five major phases: Security Hardening, ML Pipeline Completion, Performance Optimization, Testing & Quality, and DevOps & Monitoring. Tasks are organized to enable incremental progress with early validation through checkpoints.

## Tasks

- [ ] 1. Security Hardening - Authentication & Session Management
  - [ ] 1.1 Remove localStorage authentication fallback from ProtectedRoute component
    - Update `frontend/src/components/ProtectedRoute.tsx` to remove localStorage check
    - Implement server-side session validation only
    - Add redirect to /auth on invalid session
    - _Requirements: 1.1, 1.2_
  
  - [ ] 1.2 Implement secure cookie configuration for Supabase Auth
    - Install @supabase/ssr package for Next.js SSR support
    - Create server-side Supabase client with cookie handling
    - Configure httpOnly, secure, sameSite=strict cookie flags
    - Update authentication flow to use server-side validation
    - _Requirements: 1.4_
  
  - [ ] 1.3 Add CSRF protection middleware
    - Create CSRF token generation and validation middleware
    - Add CSRF token to all state-changing forms
    - Validate CSRF tokens on POST/PUT/DELETE/PATCH requests
    - _Requirements: 1.3_
  
  - [ ]* 1.4 Write property test for session validation
    - **Property 1: Session Validation Consistency**
    - **Validates: Requirements 1.1, 1.2**
  
  - [ ]* 1.5 Write property test for CSRF protection
    - **Property 2: CSRF Protection Universality**
    - **Validates: Requirements 1.3**
  
  - [ ]* 1.6 Write property test for authentication logging
    - **Property 3: Authentication Failure Logging**
    - **Validates: Requirements 1.5**

- [ ] 2. Security Hardening - File Upload Validation
  - [x] 2.1 Create FileValidator class with comprehensive validation
    - Create `backend/app/core/file_validator.py`
    - Implement size validation (10MB limit)
    - Implement MIME type validation (JPEG, PNG, WebP only)
    - Implement content verification using python-magic
    - Implement filename sanitization
    - Add detailed error responses with error codes
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_
  
  - [x] 2.2 Integrate FileValidator into all upload endpoints
    - Update `/api/v1/process-tryon` endpoint
    - Update `/api/v1/recommend` endpoint
    - Update `/api/v1/analyze-image` endpoint
    - Update `/api/v1/combine-head-body` endpoint
    - Update `/api/v1/generate-bodies` endpoint
    - _Requirements: 2.1, 2.2, 2.3, 2.4_
  
  - [ ]* 2.3 Write property tests for file validation
    - **Property 4: File Size Enforcement**
    - **Property 5: MIME Type Validation**
    - **Property 6: Content Type Verification**
    - **Property 7: Filename Sanitization**
    - **Property 8: Error Message Safety**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

- [ ] 3. Security Hardening - Rate Limiting
  - [ ] 3.1 Set up Redis for distributed rate limiting
    - Add Redis to docker-compose.yml
    - Add redis-py to requirements.txt
    - Create Redis connection pool configuration
    - _Requirements: 3.1_
  
  - [ ] 3.2 Implement RateLimiter class with token bucket algorithm
    - Create `backend/app/core/rate_limiter.py`
    - Implement token bucket algorithm using Redis
    - Support per-user and per-IP rate limiting
    - Configure different limits per endpoint type
    - Return HTTP 429 with Retry-After header
    - Add rate limit headers to responses
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
  
  - [ ] 3.3 Add rate limiting middleware to FastAPI
    - Create rate limiting middleware
    - Apply to all API endpoints with appropriate limits
    - ML endpoints: 10 requests/minute
    - Read-only endpoints: 100 requests/minute
    - Log rate limit violations
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.6_
  
  - [ ]* 3.4 Write property tests for rate limiting
    - **Property 9: Token Bucket Rate Limiting**
    - **Property 10: Rate Limit Response Format**
    - **Property 11: Per-Endpoint Rate Limits**
    - **Property 12: Rate Limit Isolation**
    - **Property 13: Rate Limit Violation Logging**
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**

- [ ] 4. Security Hardening - HTTPS & Transport Security
  - [ ] 4.1 Implement HTTPS enforcement for production
    - Add environment detection (development/staging/production)
    - Add HTTPS redirect middleware for production
    - Add HSTS headers with max-age=31536000
    - _Requirements: 4.1, 4.2_
  
  - [ ] 4.2 Implement strict CORS validation
    - Update CORS middleware to reject wildcards in production
    - Add startup validation for CORS configuration
    - Refuse to start if wildcards configured in production
    - Log critical warnings for CORS misconfigurations
    - _Requirements: 4.3, 4.4_
  
  - [ ]* 4.3 Write integration tests for transport security
    - Test HTTPS enforcement in production mode
    - Test HSTS header presence
    - Test CORS configuration validation
    - Test startup failure on wildcard CORS in production
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ] 5. Checkpoint - Security validation
  - Ensure all security tests pass
  - Verify authentication works without localStorage
  - Verify file uploads are validated
  - Verify rate limiting is enforced
  - Ask the user if questions arise

- [x] 6. ML Pipeline - IDM-VTON Integration
  - [x] 6.1 Research and download IDM-VTON model weights
    - Clone IDM-VTON repository (https://github.com/yisol/IDM-VTON)
    - Download pre-trained model weights
    - Store weights in `backend/ml_engine/weights/idm-vton/`
    - Document model version and source
    - _Requirements: 5.1_
  
  - [x] 6.2 Implement IDM-VTON pipeline
    - Create `backend/ml_engine/pipelines/idm_vton.py`
    - Implement preprocessing (resize to 512x768, normalize)
    - Implement segmentation step
    - Implement pose estimation step
    - Implement garment transfer using IDM-VTON
    - Implement postprocessing (upscale, refinements)
    - _Requirements: 5.1, 5.2_
  
  - [x] 6.3 Update try-on service to use IDM-VTON
    - Create `backend/app/services/tryon_service.py`
    - Replace mock implementation in `/api/v1/process-tryon`
    - Integrate IDM-VTON pipeline
    - Save results to data/results/
    - Return result URL and metadata
    - _Requirements: 5.2, 5.3_
  
  - [x] 6.4 Implement intermediate result caching
    - Cache segmentation masks by image hash
    - Cache pose keypoints by image hash
    - Implement cache lookup before processing
    - Store cache in Redis with 24-hour TTL
    - _Requirements: 5.6_
  
  - [ ]* 6.5 Write property tests for try-on pipeline
    - **Property 14: Try-On Pipeline Completeness**
    - **Property 15: Try-On Output Resolution**
    - **Property 17: Intermediate Result Caching**
    - **Validates: Requirements 5.2, 5.3, 5.6**

- [x] 7. ML Pipeline - Recommendation Engine Completion
  - [x] 7.1 Implement complete image collage creation
    - Update `backend/app/services/image_collage.py`
    - Create grid layout for user photo + wardrobe + generated images
    - Resize images to consistent dimensions
    - Add labels/borders for clarity
    - _Requirements: 6.2_
  
  - [x] 7.2 Complete Gemini Vision API integration
    - Update `backend/app/services/recommendation.py`
    - Implement collage upload to Gemini Vision
    - Create prompt for style analysis with color theory
    - Parse and extract fashion keywords from response
    - Implement error handling and retries
    - _Requirements: 6.1, 6.3_
  
  - [x] 7.3 Implement eBay API product search
    - Add eBay API client configuration
    - Implement product search using extracted keywords
    - Parse and format product results
    - Rank products by relevance
    - Return product recommendations with URLs
    - _Requirements: 6.4_
  
  - [x] 7.4 Implement recommendation caching and fallback
    - Cache recommendations per user with 1-hour TTL
    - Implement circuit breaker for Gemini API
    - Implement circuit breaker for eBay API
    - Return cached results on API failures
    - Return graceful error messages
    - _Requirements: 6.5_
  
  - [x] 7.5 Implement connection pooling for external APIs
    - Configure httpx AsyncClient with connection pooling
    - Set appropriate timeout values
    - Implement retry logic with exponential backoff
    - _Requirements: 6.6_
  
  - [ ]* 7.6 Write property tests for recommendation engine
    - **Property 18: Collage Creation Consistency**
    - **Property 19: Keyword Extraction**
    - **Property 20: Product Search Integration**
    - **Property 21: API Failure Fallback**
    - **Validates: Requirements 6.2, 6.3, 6.4, 6.5**

- [ ] 8. Checkpoint - ML pipeline validation
  - Ensure try-on produces realistic results
  - Verify recommendations return relevant products
  - Test with various input images
  - Verify caching improves performance
  - Ask the user if questions arise

- [ ] 9. Performance - Model Loader Optimization
  - [x] 9.1 Implement thread-safe singleton ModelLoader
    - Update `backend/ml_engine/loader.py`
    - Implement singleton pattern with threading.Lock
    - Ensure only one instance created across threads
    - _Requirements: 8.1_
  
  - [x] 9.2 Implement model caching with LRU eviction
    - Add in-memory model cache with LRU policy
    - Track model access times
    - Implement eviction when cache full
    - Monitor GPU memory usage
    - Evict LRU models when memory >80%
    - _Requirements: 8.2, 8.3_
  
  - [x] 9.3 Implement model preloading on startup
    - Add preload_models() method to ModelLoader
    - Call during FastAPI lifespan startup
    - Preload IDM-VTON and SDXL models
    - Log loading progress and times
    - _Requirements: 7.1_
  
  - [x] 9.4 Implement model warmup
    - Create sample inputs for each model
    - Run warmup inference before accepting traffic
    - Log warmup completion
    - _Requirements: 7.2_
  
  - [ ]* 9.5 Write property tests for model loader
    - **Property 25: Singleton Thread Safety**
    - **Property 26: LRU Cache Eviction**
    - **Validates: Requirements 8.1, 8.2**

- [ ] 10. Performance - Model Quantization
  - [ ] 10.1 Research quantization for IDM-VTON and SDXL
    - Identify quantization-compatible layers
    - Test INT8 quantization on non-critical layers
    - Measure accuracy loss
    - Document quantization strategy
    - _Requirements: 7.4_
  
  - [ ] 10.2 Implement INT8 quantization
    - Apply quantization to compatible models
    - Validate accuracy loss <2%
    - Update model loading to use quantized versions
    - Add configuration flag for quantization
    - _Requirements: 7.4_
  
  - [ ]* 10.3 Write property test for quantization accuracy
    - **Property 22: Quantization Accuracy Preservation**
    - **Validates: Requirements 7.4**

- [ ] 11. Performance - Async Batching
  - [ ] 11.1 Implement async batching for inference
    - Create batching queue with configurable size (4) and wait time (2s)
    - Collect concurrent requests into batches
    - Process batches together
    - Return individual results to requesters
    - _Requirements: 7.5_
  
  - [ ] 11.2 Implement GPU memory management
    - Monitor GPU memory during inference
    - Implement automatic cleanup after inference
    - Queue requests when GPU memory exhausted
    - Process queue when memory available
    - _Requirements: 7.6, 7.7_
  
  - [ ]* 11.3 Write property tests for batching and memory
    - **Property 23: Async Batch Formation**
    - **Property 24: GPU Memory Cleanup**
    - **Validates: Requirements 7.5, 7.6**

- [ ] 12. Performance - Response Optimization
  - [ ] 12.1 Implement HTTP response caching
    - Add response caching middleware
    - Cache identical requests with 1-hour TTL
    - Use request hash as cache key
    - Return cached responses with cache headers
    - _Requirements: 16.3_
  
  - [ ] 12.2 Implement response compression
    - Add gzip compression middleware
    - Compress responses >1KB
    - Add appropriate content-encoding headers
    - _Requirements: 16.6_
  
  - [ ]* 12.3 Write property tests for response optimization
    - **Property 41: Response Caching**
    - **Property 42: Response Compression**
    - **Validates: Requirements 16.3, 16.6**

- [ ] 13. Checkpoint - Performance validation
  - Measure health check response time (<100ms)
  - Measure try-on inference time (<10s)
  - Verify caching improves response times
  - Monitor GPU memory usage
  - Ask the user if questions arise

- [ ] 14. Testing - Input Validation Tests
  - [x] 14.1 Update Pydantic schemas with comprehensive validation
    - Update `backend/app/models/schemas.py`
    - Add validators for all numeric ranges
    - Add validators for enum values
    - Add validators for string formats
    - Configure strict schema validation (reject extra fields)
    - _Requirements: 10.1, 10.4, 10.5, 10.6_
  
  - [ ] 14.2 Implement XSS sanitization
    - Add bleach library for HTML sanitization
    - Create sanitization utility functions
    - Apply to all user-provided strings
    - _Requirements: 10.3_
  
  - [ ]* 14.3 Write property tests for input validation
    - **Property 27: Pydantic Validation Enforcement**
    - **Property 28: XSS Sanitization**
    - **Property 29: Numeric Range Validation**
    - **Property 30: Enum Validation**
    - **Property 31: Strict Schema Validation**
    - **Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5, 10.6**

- [ ] 15. Testing - Error Handling & Logging
  - [x] 15.1 Implement structured JSON logging
    - Update `backend/app/core/logging_config.py`
    - Configure JSON formatter with required fields
    - Add request ID generation and propagation
    - Ensure all logs include request_id
    - _Requirements: 11.1, 11.5_
  
  - [x] 15.2 Implement custom exception classes
    - Create `backend/app/core/exceptions.py`
    - Define exception classes for each error category
    - Map exceptions to error codes
    - _Requirements: 11.3_
  
  - [ ] 15.3 Implement consistent error response format
    - Create error response middleware
    - Format all errors using ErrorResponse schema
    - Include error_code, message, details, request_id
    - Log all errors with stack traces
    - _Requirements: 11.2, 11.4_
  
  - [ ] 15.4 Add performance metrics logging
    - Log inference duration for all ML operations
    - Log model name, batch size, GPU memory
    - Log cache hit/miss events
    - _Requirements: 11.6_
  
  - [ ]* 15.5 Write property tests for logging and errors
    - **Property 32: Structured Log Format**
    - **Property 33: Error Stack Trace Logging**
    - **Property 34: Consistent Error Response Format**
    - **Property 35: Request ID Propagation**
    - **Property 36: Inference Metrics Logging**
    - **Validates: Requirements 11.1, 11.2, 11.4, 11.5, 11.6**

- [ ] 16. Testing - Configuration Management
  - [ ] 16.1 Implement configuration validation
    - Create `backend/app/core/config.py`
    - Define Pydantic models for all configuration
    - Validate required environment variables at startup
    - Fail fast with clear errors on missing config
    - _Requirements: 15.1, 15.2, 15.3_
  
  - [ ] 16.2 Implement sensitive value masking
    - Create masking utility for API keys and passwords
    - Apply to all log entries
    - Apply to all error messages
    - _Requirements: 15.5_
  
  - [ ]* 16.3 Write property tests for configuration
    - **Property 37: Configuration Type Validation**
    - **Property 38: Sensitive Value Masking**
    - **Validates: Requirements 15.3, 15.5**

- [ ] 17. Testing - Unit Test Suite
  - [ ] 17.1 Set up testing infrastructure
    - Add pytest, pytest-asyncio, pytest-cov to requirements.txt
    - Create `backend/tests/` directory structure
    - Configure pytest.ini with coverage settings
    - Create test fixtures and utilities
    - _Requirements: 9.1, 9.2_
  
  - [ ] 17.2 Write unit tests for service layer
    - Test FileValidator methods
    - Test RateLimiter methods
    - Test ModelLoader methods
    - Test recommendation service methods
    - Test try-on service methods
    - Target 80% code coverage
    - _Requirements: 9.2_
  
  - [ ]* 17.3 Write integration tests for API endpoints
    - Test authentication flows
    - Test file upload validation
    - Test rate limiting enforcement
    - Test error response formats
    - Mock ML models for fast tests
    - _Requirements: 9.4_

- [ ] 18. Testing - Property-Based Test Suite
  - [ ] 18.1 Set up Hypothesis for property testing
    - Add hypothesis to requirements.txt
    - Configure hypothesis settings (100 iterations minimum)
    - Create property test utilities and strategies
    - _Requirements: 9.3_
  
  - [ ] 18.2 Implement all property tests from design
    - Implement all 44 properties defined in design document
    - Tag each test with feature name and property number
    - Ensure tests reference design document properties
    - _Requirements: 9.3_

- [ ] 19. Checkpoint - Testing validation
  - Run full test suite
  - Verify 80% code coverage achieved
  - Verify all property tests pass
  - Fix any failing tests
  - Ask the user if questions arise

- [ ] 20. DevOps - Docker Optimization
  - [ ] 20.1 Implement multi-stage Docker build
    - Update `backend/Dockerfile`
    - Create builder stage for dependencies
    - Create runtime stage with minimal image
    - Copy only necessary files to runtime
    - _Requirements: 12.1_
  
  - [ ] 20.2 Add NVIDIA GPU support to Dockerfile
    - Use NVIDIA CUDA base image
    - Install CUDA toolkit and cuDNN
    - Configure GPU device access
    - _Requirements: 12.2_
  
  - [ ] 20.3 Update docker-compose with resource limits
    - Add CPU limits
    - Add memory limits
    - Add GPU device reservations
    - Add health checks
    - Add volume mounts for persistent data
    - _Requirements: 12.3, 12.4, 12.6_
  
  - [ ] 20.4 Implement graceful shutdown
    - Add signal handlers for SIGTERM
    - Complete in-flight requests before shutdown
    - Close database connections
    - Unload ML models
    - _Requirements: 12.5_

- [ ] 21. DevOps - Health Monitoring
  - [ ] 21.1 Implement comprehensive health checks
    - Create `backend/app/core/health.py`
    - Check GPU availability and memory
    - Check model loading status
    - Check external API connectivity (Gemini, eBay, Supabase)
    - Check Redis connectivity
    - Return health status (healthy/degraded/unhealthy)
    - _Requirements: 13.1, 13.2, 13.3, 13.5_
  
  - [ ] 21.2 Implement Prometheus metrics endpoint
    - Add prometheus-client to requirements.txt
    - Expose /metrics endpoint
    - Track request count, latency, errors per endpoint
    - Track model inference time and throughput
    - Track GPU memory usage
    - Track cache hit rates
    - Track rate limit violations
    - _Requirements: 13.4_
  
  - [ ]* 21.3 Write integration tests for health and metrics
    - Test health check endpoint returns correct status
    - Test metrics endpoint returns Prometheus format
    - Test degraded status on dependency failures
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5_

- [ ] 22. DevOps - API Documentation
  - [ ] 22.1 Implement API versioning
    - Ensure all endpoints use /api/v1 prefix
    - Document versioning strategy
    - _Requirements: 14.1_
  
  - [ ] 22.2 Configure OpenAPI documentation
    - Update FastAPI app with comprehensive metadata
    - Add descriptions to all endpoints
    - Add request/response examples
    - Add error code documentation
    - Verify /docs endpoint serves interactive documentation
    - _Requirements: 14.2, 14.3, 14.4, 14.5_
  
  - [ ]* 22.3 Write integration tests for API documentation
    - Test /docs endpoint accessibility
    - Test OpenAPI spec generation
    - Validate OpenAPI spec format
    - _Requirements: 14.1, 14.2, 14.3_

- [ ] 23. DevOps - Development Workflow
  - [ ] 23.1 Update development startup script
    - Update `dev_start.ps1`
    - Add dependency checking (Python, Node, Redis)
    - Add clear error messages for missing dependencies
    - Implement cleanup on failure
    - _Requirements: 17.1, 17.5, 17.6_
  
  - [ ] 23.2 Create mock services for development
    - Create mock Gemini API responses
    - Create mock eBay API responses
    - Add development mode flag
    - _Requirements: 17.4_
  
  - [ ] 23.3 Set up code quality tools
    - Add ruff, black, mypy to requirements.txt
    - Create pyproject.toml with tool configurations
    - Add pre-commit hooks
    - Add npm scripts for linting and formatting
    - _Requirements: 18.1, 18.2, 18.3, 18.4_

- [ ] 24. DevOps - CI/CD Pipeline
  - [ ] 24.1 Create GitHub Actions workflow
    - Create `.github/workflows/ci.yml`
    - Run linting and type checking on every commit
    - Run unit tests on every commit
    - Run property tests on every commit
    - Run integration tests on every PR
    - Check code coverage (fail if <80%)
    - _Requirements: 9.6_
  
  - [ ] 24.2 Set up security scanning
    - Add dependency vulnerability scanning
    - Add Docker image scanning
    - Add secret scanning
    - Block deployment on critical vulnerabilities
    - _Requirements: 19.1, 19.3, 19.5, 19.6_

- [ ] 25. Final Checkpoint - Production readiness
  - Run full test suite (unit, property, integration)
  - Verify all health checks pass
  - Verify metrics are exposed
  - Test Docker deployment
  - Review security checklist
  - Verify documentation is complete
  - Ask the user if questions arise

- [ ] 26. Deployment Preparation
  - [ ] 26.1 Create deployment documentation
    - Document environment variables
    - Document deployment steps
    - Document rollback procedures
    - Document monitoring and alerting setup
    - _Requirements: 15.4_
  
  - [ ] 26.2 Create production configuration
    - Create production .env.example
    - Document API key rotation strategy
    - Configure production CORS origins
    - Enable HTTPS enforcement
    - _Requirements: 4.1, 4.3, 19.4_
  
  - [ ] 26.3 Set up monitoring and alerting
    - Configure Prometheus scraping
    - Create Grafana dashboards
    - Set up alerts for error rates
    - Set up alerts for GPU memory
    - Set up alerts for health check failures
    - _Requirements: 13.6_

## Notes

- Tasks marked with `*` are optional property-based tests that can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at key milestones
- Property tests validate universal correctness properties with 100+ iterations
- Integration tests validate API contracts with mocked dependencies
- The implementation follows a phased approach: Security → ML → Performance → Testing → DevOps
