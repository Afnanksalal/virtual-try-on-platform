-- ============================================
-- Virtual Try-On Platform - Supabase Setup
-- ============================================
-- Run this script in Supabase SQL Editor
-- This will create all necessary tables, storage buckets, and policies

-- ============================================
-- CLEANUP (Optional - only if you want fresh start)
-- ============================================
-- Uncomment these lines if you want to drop everything and start fresh
-- WARNING: This will delete all data!

-- DROP POLICY IF EXISTS "Users can view own profile" ON profiles;
-- DROP POLICY IF EXISTS "Users can insert own profile" ON profiles;
-- DROP POLICY IF EXISTS "Users can update own profile" ON profiles;
-- DROP POLICY IF EXISTS "Users can delete own profile" ON profiles;
-- DROP TABLE IF EXISTS profiles CASCADE;

-- DROP POLICY IF EXISTS "Users can view own wardrobe" ON wardrobe;
-- DROP POLICY IF EXISTS "Users can insert own wardrobe items" ON wardrobe;
-- DROP POLICY IF EXISTS "Users can update own wardrobe items" ON wardrobe;
-- DROP POLICY IF EXISTS "Users can delete own wardrobe items" ON wardrobe;
-- DROP TABLE IF EXISTS wardrobe CASCADE;

-- DROP POLICY IF EXISTS "Users can view own try-on history" ON tryon_history;
-- DROP POLICY IF EXISTS "Users can insert own try-on history" ON tryon_history;
-- DROP POLICY IF EXISTS "Users can delete own try-on history" ON tryon_history;
-- DROP TABLE IF EXISTS tryon_history CASCADE;

-- DROP POLICY IF EXISTS "Users can view own generated bodies" ON generated_bodies;
-- DROP POLICY IF EXISTS "Users can insert own generated bodies" ON generated_bodies;
-- DROP POLICY IF EXISTS "Users can update own generated bodies" ON generated_bodies;
-- DROP POLICY IF EXISTS "Users can delete own generated bodies" ON generated_bodies;
-- DROP TABLE IF EXISTS generated_bodies CASCADE;

-- ============================================
-- 1. PROFILES TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  email TEXT NOT NULL,
  name TEXT NOT NULL,
  age INTEGER NOT NULL CHECK (age >= 13 AND age <= 100),
  gender TEXT NOT NULL CHECK (gender IN ('male', 'female', 'other')),
  height_cm DECIMAL(5,2) NOT NULL CHECK (height_cm >= 50 AND height_cm <= 300),
  weight_kg DECIMAL(5,2) NOT NULL CHECK (weight_kg >= 20 AND weight_kg <= 500),
  photo_url TEXT,
  ethnicity TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist, then recreate
DO $$ 
BEGIN
  DROP POLICY IF EXISTS "Users can view own profile" ON profiles;
  DROP POLICY IF EXISTS "Users can insert own profile" ON profiles;
  DROP POLICY IF EXISTS "Users can update own profile" ON profiles;
  DROP POLICY IF EXISTS "Users can delete own profile" ON profiles;
END $$;

-- Policies for profiles table
CREATE POLICY "Users can view own profile"
  ON profiles FOR SELECT
  USING (auth.uid() = id);

CREATE POLICY "Users can insert own profile"
  ON profiles FOR INSERT
  WITH CHECK (auth.uid() = id);

CREATE POLICY "Users can update own profile"
  ON profiles FOR UPDATE
  USING (auth.uid() = id)
  WITH CHECK (auth.uid() = id);

CREATE POLICY "Users can delete own profile"
  ON profiles FOR DELETE
  USING (auth.uid() = id);

-- ============================================
-- 2. WARDROBE TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS wardrobe (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  item_name TEXT NOT NULL,
  category TEXT NOT NULL CHECK (category IN ('top', 'bottom', 'dress', 'outerwear', 'shoes', 'accessories')),
  image_url TEXT NOT NULL,
  brand TEXT,
  color TEXT,
  size TEXT,
  purchase_date DATE,
  notes TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE wardrobe ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist, then recreate
DO $$ 
BEGIN
  DROP POLICY IF EXISTS "Users can view own wardrobe" ON wardrobe;
  DROP POLICY IF EXISTS "Users can insert own wardrobe items" ON wardrobe;
  DROP POLICY IF EXISTS "Users can update own wardrobe items" ON wardrobe;
  DROP POLICY IF EXISTS "Users can delete own wardrobe items" ON wardrobe;
END $$;

-- Policies for wardrobe table
CREATE POLICY "Users can view own wardrobe"
  ON wardrobe FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own wardrobe items"
  ON wardrobe FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own wardrobe items"
  ON wardrobe FOR UPDATE
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own wardrobe items"
  ON wardrobe FOR DELETE
  USING (auth.uid() = user_id);

-- ============================================
-- 3. TRY-ON HISTORY TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS tryon_history (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  user_image_url TEXT NOT NULL,
  garment_image_url TEXT NOT NULL,
  result_image_url TEXT NOT NULL,
  processing_time_seconds DECIMAL(10,2),
  status TEXT NOT NULL CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
  error_message TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE tryon_history ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist, then recreate
DO $$ 
BEGIN
  DROP POLICY IF EXISTS "Users can view own try-on history" ON tryon_history;
  DROP POLICY IF EXISTS "Users can insert own try-on history" ON tryon_history;
  DROP POLICY IF EXISTS "Users can delete own try-on history" ON tryon_history;
END $$;

-- Policies for tryon_history table
CREATE POLICY "Users can view own try-on history"
  ON tryon_history FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own try-on history"
  ON tryon_history FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own try-on history"
  ON tryon_history FOR DELETE
  USING (auth.uid() = user_id);

-- ============================================
-- 4. GENERATED BODIES TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS generated_bodies (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  image_url TEXT NOT NULL,
  ethnicity TEXT,
  skin_tone TEXT,
  body_type TEXT NOT NULL CHECK (body_type IN ('athletic', 'slim', 'muscular', 'average', 'curvy', 'plus_size')),
  height_cm DECIMAL(5,2),
  weight_kg DECIMAL(5,2),
  is_selected BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE generated_bodies ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist, then recreate
DO $$ 
BEGIN
  DROP POLICY IF EXISTS "Users can view own generated bodies" ON generated_bodies;
  DROP POLICY IF EXISTS "Users can insert own generated bodies" ON generated_bodies;
  DROP POLICY IF EXISTS "Users can update own generated bodies" ON generated_bodies;
  DROP POLICY IF EXISTS "Users can delete own generated bodies" ON generated_bodies;
END $$;

-- Policies for generated_bodies table
CREATE POLICY "Users can view own generated bodies"
  ON generated_bodies FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own generated bodies"
  ON generated_bodies FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own generated bodies"
  ON generated_bodies FOR UPDATE
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own generated bodies"
  ON generated_bodies FOR DELETE
  USING (auth.uid() = user_id);

-- ============================================
-- 5. INDEXES FOR PERFORMANCE
-- ============================================
CREATE INDEX IF NOT EXISTS idx_wardrobe_user_id ON wardrobe(user_id);
CREATE INDEX IF NOT EXISTS idx_wardrobe_category ON wardrobe(category);
CREATE INDEX IF NOT EXISTS idx_tryon_history_user_id ON tryon_history(user_id);
CREATE INDEX IF NOT EXISTS idx_tryon_history_created_at ON tryon_history(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_generated_bodies_user_id ON generated_bodies(user_id);
CREATE INDEX IF NOT EXISTS idx_generated_bodies_is_selected ON generated_bodies(is_selected);

-- ============================================
-- 6. UPDATED_AT TRIGGER FUNCTION
-- ============================================
-- Function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to profiles table
DROP TRIGGER IF EXISTS update_profiles_updated_at ON profiles;
CREATE TRIGGER update_profiles_updated_at
  BEFORE UPDATE ON profiles
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- Apply trigger to wardrobe table
DROP TRIGGER IF EXISTS update_wardrobe_updated_at ON wardrobe;
CREATE TRIGGER update_wardrobe_updated_at
  BEFORE UPDATE ON wardrobe
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- 7. STORAGE BUCKETS SETUP
-- ============================================
-- Create storage buckets (will skip if already exist)
INSERT INTO storage.buckets (id, name, public)
VALUES ('user-uploads', 'user-uploads', true)
ON CONFLICT (id) DO UPDATE SET public = true;

INSERT INTO storage.buckets (id, name, public)
VALUES ('generated-bodies', 'generated-bodies', true)
ON CONFLICT (id) DO UPDATE SET public = true;

INSERT INTO storage.buckets (id, name, public)
VALUES ('tryon-results', 'tryon-results', true)
ON CONFLICT (id) DO UPDATE SET public = true;

-- ============================================
-- 8. STORAGE POLICIES
-- ============================================

-- Drop existing storage policies if they exist
DO $$ 
BEGIN
  -- user-uploads policies
  DROP POLICY IF EXISTS "Users can upload own files" ON storage.objects;
  DROP POLICY IF EXISTS "Users can view own files" ON storage.objects;
  DROP POLICY IF EXISTS "Users can update own files" ON storage.objects;
  DROP POLICY IF EXISTS "Users can delete own files" ON storage.objects;
  DROP POLICY IF EXISTS "Public can view user uploads" ON storage.objects;
  
  -- generated-bodies policies
  DROP POLICY IF EXISTS "Users can upload generated bodies" ON storage.objects;
  DROP POLICY IF EXISTS "Users can view generated bodies" ON storage.objects;
  DROP POLICY IF EXISTS "Users can delete generated bodies" ON storage.objects;
  DROP POLICY IF EXISTS "Public can view generated bodies" ON storage.objects;
  
  -- tryon-results policies
  DROP POLICY IF EXISTS "Users can upload tryon results" ON storage.objects;
  DROP POLICY IF EXISTS "Users can view tryon results" ON storage.objects;
  DROP POLICY IF EXISTS "Users can delete tryon results" ON storage.objects;
  DROP POLICY IF EXISTS "Public can view tryon results" ON storage.objects;
END $$;

-- user-uploads bucket policies
CREATE POLICY "Users can upload own files"
  ON storage.objects FOR INSERT
  TO authenticated
  WITH CHECK (bucket_id = 'user-uploads' AND (storage.foldername(name))[1] = auth.uid()::text);

CREATE POLICY "Users can view own files"
  ON storage.objects FOR SELECT
  TO authenticated
  USING (bucket_id = 'user-uploads' AND (storage.foldername(name))[1] = auth.uid()::text);

CREATE POLICY "Users can update own files"
  ON storage.objects FOR UPDATE
  TO authenticated
  USING (bucket_id = 'user-uploads' AND (storage.foldername(name))[1] = auth.uid()::text);

CREATE POLICY "Users can delete own files"
  ON storage.objects FOR DELETE
  TO authenticated
  USING (bucket_id = 'user-uploads' AND (storage.foldername(name))[1] = auth.uid()::text);

CREATE POLICY "Public can view user uploads"
  ON storage.objects FOR SELECT
  TO public
  USING (bucket_id = 'user-uploads');

-- generated-bodies bucket policies
CREATE POLICY "Users can upload generated bodies"
  ON storage.objects FOR INSERT
  TO authenticated
  WITH CHECK (bucket_id = 'generated-bodies' AND (storage.foldername(name))[1] = auth.uid()::text);

CREATE POLICY "Users can view generated bodies"
  ON storage.objects FOR SELECT
  TO authenticated
  USING (bucket_id = 'generated-bodies' AND (storage.foldername(name))[1] = auth.uid()::text);

CREATE POLICY "Users can delete generated bodies"
  ON storage.objects FOR DELETE
  TO authenticated
  USING (bucket_id = 'generated-bodies' AND (storage.foldername(name))[1] = auth.uid()::text);

CREATE POLICY "Public can view generated bodies"
  ON storage.objects FOR SELECT
  TO public
  USING (bucket_id = 'generated-bodies');

-- tryon-results bucket policies
CREATE POLICY "Users can upload tryon results"
  ON storage.objects FOR INSERT
  TO authenticated
  WITH CHECK (bucket_id = 'tryon-results' AND (storage.foldername(name))[1] = auth.uid()::text);

CREATE POLICY "Users can view tryon results"
  ON storage.objects FOR SELECT
  TO authenticated
  USING (bucket_id = 'tryon-results' AND (storage.foldername(name))[1] = auth.uid()::text);

CREATE POLICY "Users can delete tryon results"
  ON storage.objects FOR DELETE
  TO authenticated
  USING (bucket_id = 'tryon-results' AND (storage.foldername(name))[1] = auth.uid()::text);

CREATE POLICY "Public can view tryon results"
  ON storage.objects FOR SELECT
  TO public
  USING (bucket_id = 'tryon-results');

-- ============================================
-- 9. VERIFICATION QUERIES
-- ============================================
-- Run these to verify everything is set up correctly

-- Check tables
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
  AND table_name IN ('profiles', 'wardrobe', 'tryon_history', 'generated_bodies');

-- Check RLS is enabled
SELECT tablename, rowsecurity 
FROM pg_tables 
WHERE schemaname = 'public' 
  AND tablename IN ('profiles', 'wardrobe', 'tryon_history', 'generated_bodies');

-- Check storage buckets
SELECT * FROM storage.buckets 
WHERE name IN ('user-uploads', 'generated-bodies', 'tryon-results');

-- ============================================
-- SETUP COMPLETE!
-- ============================================
-- Next steps:
-- 1. Verify all tables exist
-- 2. Verify storage buckets are created
-- 3. Test authentication and profile creation
-- 4. Upload test images to verify storage policies
