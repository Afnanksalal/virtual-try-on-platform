# Product Overview

Virtual Try-On Platform - An AI-powered fashion technology application that enables users to virtually try on clothing using machine learning models.

## Core Features

- **Virtual Try-On**: Upload user photos and garment images to generate realistic try-on results using ML models (segmentation, pose estimation, IDM-VTON)
- **Body Generation**: Create synthetic body models based on user parameters (ethnicity, body type, height, weight) using SDXL diffusion models
- **AI Recommendations**: Get personalized outfit recommendations using Gemini Vision API and eBay product search
- **Smart Onboarding**: Analyze uploaded photos to determine if they're head-only or full-body shots, then generate appropriate body models
- **Image Composition**: Combine head shots with generated body models for complete virtual try-on experiences
- **3D Visualization**: Future support for 3D model reconstruction (PIFuHD)

## User Flow

1. User uploads their photo during onboarding
2. System analyzes if photo is head-only or full-body
3. If head-only, generates body models and combines with user's face
4. User can then try on garments in the Studio
5. AI provides outfit recommendations based on user's style
6. Results can be downloaded or used for shopping

## Authentication

Uses Supabase for user authentication and session management.
