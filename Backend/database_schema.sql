-- =====================================================================
-- ZeroDay Face Recognition Authentication Database Schema
-- Run this SQL in your Supabase SQL Editor
-- =====================================================================

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =====================================================================
-- 1. PROFILES TABLE
-- =====================================================================
CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    full_name VARCHAR(100) NOT NULL,
    email VARCHAR(255) NOT NULL,
    phone VARCHAR(20),
    face_registered BOOLEAN DEFAULT FALSE,
    registration_completed BOOLEAN DEFAULT FALSE,
    account_status VARCHAR(20) DEFAULT 'active' CHECK (account_status IN ('active', 'suspended', 'pending', 'deactivated')),
    two_factor_enabled BOOLEAN DEFAULT FALSE,
    last_login_at TIMESTAMPTZ,
    failed_login_attempts INTEGER DEFAULT 0,
    last_failed_login_at TIMESTAMPTZ,
    account_locked_until TIMESTAMPTZ,
    profile_image_url TEXT,
    preferences JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- =====================================================================
-- 2. FACE EMBEDDINGS TABLE
-- =====================================================================
CREATE TABLE IF NOT EXISTS public.face_embeddings (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE NOT NULL,
    embedding_data JSONB NOT NULL, -- Store the face embedding vector
    image_number INTEGER NOT NULL, -- 1, 2, or 3 for the three registration images
    confidence_score FLOAT NOT NULL,
    anti_spoofing_passed BOOLEAN NOT NULL,
    liveness_score FLOAT,
    image_path TEXT, -- Path to stored image in Supabase storage
    model_config JSONB DEFAULT '{}', -- Store model parameters used
    embedding_version VARCHAR(10) DEFAULT 'v1.0',
    quality_score FLOAT,
    detection_confidence FLOAT,
    face_area JSONB, -- Store bounding box coordinates
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Ensure one embedding per image number per user
    UNIQUE(user_id, image_number)
);

-- =====================================================================
-- 3. FACE RECOGNITION LOGS TABLE
-- =====================================================================
CREATE TABLE IF NOT EXISTS public.face_recognition_logs (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE,
    session_id VARCHAR(100), -- Optional session identifier
    activity_type VARCHAR(50) NOT NULL CHECK (activity_type IN (
        'registration', 'login_attempt', 'liveness_check', 
        'anti_spoofing', 'face_update', 'verification'
    )),
    success BOOLEAN NOT NULL,
    confidence_score FLOAT,
    anti_spoofing_result JSONB,
    liveness_result JSONB,
    details JSONB DEFAULT '{}',
    error_message TEXT,
    processing_time_ms INTEGER,
    image_count INTEGER,
    
    -- Security and audit fields
    ip_address INET,
    user_agent TEXT,
    device_info JSONB DEFAULT '{}',
    location_info JSONB DEFAULT '{}',
    risk_score FLOAT,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =====================================================================
-- 4. LIVENESS SESSIONS TABLE
-- =====================================================================
CREATE TABLE IF NOT EXISTS public.liveness_sessions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    session_type VARCHAR(50) NOT NULL CHECK (session_type IN (
        'registration', 'login', 'verification', 'account_recovery'
    )),
    challenges JSONB NOT NULL, -- Array of challenge types: ["blink", "smile", "turn_left", "turn_right"]
    completed_challenges JSONB DEFAULT '[]',
    challenge_results JSONB DEFAULT '{}',
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN (
        'pending', 'in_progress', 'completed', 'failed', 'expired', 'cancelled'
    )),
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 3,
    current_challenge_index INTEGER DEFAULT 0,
    total_score FLOAT,
    individual_scores JSONB DEFAULT '{}',
    
    -- Timing controls
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ NOT NULL,
    
    -- Security
    ip_address INET,
    user_agent TEXT,
    device_fingerprint TEXT,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- =====================================================================
-- 5. AUTHENTICATION SESSIONS TABLE
-- =====================================================================
CREATE TABLE IF NOT EXISTS public.auth_sessions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE NOT NULL,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    session_type VARCHAR(50) NOT NULL CHECK (session_type IN (
        'face_login', 'password_login', 'two_factor', 'recovery'
    )),
    is_active BOOLEAN DEFAULT TRUE,
    expires_at TIMESTAMPTZ NOT NULL,
    last_activity_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Device and location tracking
    device_id VARCHAR(255),
    device_name VARCHAR(100),
    device_type VARCHAR(50),
    browser_info JSONB DEFAULT '{}',
    ip_address INET,
    location_info JSONB DEFAULT '{}',
    
    -- Security flags
    is_trusted_device BOOLEAN DEFAULT FALSE,
    risk_score FLOAT DEFAULT 0.0,
    security_flags JSONB DEFAULT '{}',
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- =====================================================================
-- 6. SECURITY EVENTS TABLE
-- =====================================================================
CREATE TABLE IF NOT EXISTS public.security_events (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE,
    event_type VARCHAR(50) NOT NULL CHECK (event_type IN (
        'suspicious_login', 'multiple_failed_attempts', 'unusual_location',
        'device_change', 'potential_spoofing', 'account_lockout',
        'password_change', 'email_change', 'face_update'
    )),
    severity VARCHAR(20) DEFAULT 'medium' CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    description TEXT NOT NULL,
    details JSONB DEFAULT '{}',
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMPTZ,
    resolved_by UUID REFERENCES public.profiles(id),
    
    -- Context information
    ip_address INET,
    user_agent TEXT,
    device_info JSONB DEFAULT '{}',
    location_info JSONB DEFAULT '{}',
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =====================================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================================

-- Profiles indexes
CREATE INDEX IF NOT EXISTS idx_profiles_username ON public.profiles(username);
CREATE INDEX IF NOT EXISTS idx_profiles_email ON public.profiles(email);
CREATE INDEX IF NOT EXISTS idx_profiles_face_registered ON public.profiles(face_registered);
CREATE INDEX IF NOT EXISTS idx_profiles_account_status ON public.profiles(account_status);

-- Face embeddings indexes
CREATE INDEX IF NOT EXISTS idx_face_embeddings_user_id ON public.face_embeddings(user_id);
CREATE INDEX IF NOT EXISTS idx_face_embeddings_is_active ON public.face_embeddings(is_active);
CREATE INDEX IF NOT EXISTS idx_face_embeddings_user_active ON public.face_embeddings(user_id, is_active);

-- Face recognition logs indexes
CREATE INDEX IF NOT EXISTS idx_face_logs_user_id ON public.face_recognition_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_face_logs_activity_type ON public.face_recognition_logs(activity_type);
CREATE INDEX IF NOT EXISTS idx_face_logs_created_at ON public.face_recognition_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_face_logs_success ON public.face_recognition_logs(success);

-- Liveness sessions indexes
CREATE INDEX IF NOT EXISTS idx_liveness_sessions_token ON public.liveness_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_liveness_sessions_user_id ON public.liveness_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_liveness_sessions_status ON public.liveness_sessions(status);
CREATE INDEX IF NOT EXISTS idx_liveness_sessions_expires_at ON public.liveness_sessions(expires_at);

-- Auth sessions indexes
CREATE INDEX IF NOT EXISTS idx_auth_sessions_user_id ON public.auth_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_auth_sessions_token ON public.auth_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_auth_sessions_active ON public.auth_sessions(is_active);
CREATE INDEX IF NOT EXISTS idx_auth_sessions_expires_at ON public.auth_sessions(expires_at);

-- Security events indexes
CREATE INDEX IF NOT EXISTS idx_security_events_user_id ON public.security_events(user_id);
CREATE INDEX IF NOT EXISTS idx_security_events_type ON public.security_events(event_type);
CREATE INDEX IF NOT EXISTS idx_security_events_severity ON public.security_events(severity);
CREATE INDEX IF NOT EXISTS idx_security_events_created_at ON public.security_events(created_at);

-- =====================================================================
-- ROW LEVEL SECURITY (RLS) POLICIES
-- =====================================================================

-- Enable RLS on all tables
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.face_embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.face_recognition_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.liveness_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.auth_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.security_events ENABLE ROW LEVEL SECURITY;

-- Profiles policies
CREATE POLICY "Users can view own profile" ON public.profiles
    FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own profile" ON public.profiles
    FOR UPDATE USING (auth.uid() = id);

CREATE POLICY "Users can insert own profile" ON public.profiles
    FOR INSERT WITH CHECK (auth.uid() = id);

-- Face embeddings policies
CREATE POLICY "Users can view own face embeddings" ON public.face_embeddings
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own face embeddings" ON public.face_embeddings
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own face embeddings" ON public.face_embeddings
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own face embeddings" ON public.face_embeddings
    FOR DELETE USING (auth.uid() = user_id);

-- Face recognition logs policies
CREATE POLICY "Users can view own face recognition logs" ON public.face_recognition_logs
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert face recognition logs" ON public.face_recognition_logs
    FOR INSERT WITH CHECK (auth.uid() = user_id OR user_id IS NULL);

-- Liveness sessions policies
CREATE POLICY "Users can view own liveness sessions" ON public.liveness_sessions
    FOR SELECT USING (auth.uid() = user_id OR user_id IS NULL);

CREATE POLICY "Users can insert liveness sessions" ON public.liveness_sessions
    FOR INSERT WITH CHECK (auth.uid() = user_id OR user_id IS NULL);

CREATE POLICY "Users can update own liveness sessions" ON public.liveness_sessions
    FOR UPDATE USING (auth.uid() = user_id OR user_id IS NULL);

-- Auth sessions policies
CREATE POLICY "Users can view own auth sessions" ON public.auth_sessions
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own auth sessions" ON public.auth_sessions
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own auth sessions" ON public.auth_sessions
    FOR UPDATE USING (auth.uid() = user_id);

-- Security events policies
CREATE POLICY "Users can view own security events" ON public.security_events
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "System can insert security events" ON public.security_events
    FOR INSERT WITH CHECK (true);

-- =====================================================================
-- FUNCTIONS AND TRIGGERS
-- =====================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
CREATE TRIGGER update_profiles_updated_at BEFORE UPDATE ON public.profiles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_liveness_sessions_updated_at BEFORE UPDATE ON public.liveness_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_auth_sessions_updated_at BEFORE UPDATE ON public.auth_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to handle user signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.profiles (id, username, full_name, email)
    VALUES (
        NEW.id,
        COALESCE(NEW.raw_user_meta_data->>'username', ''),
        COALESCE(NEW.raw_user_meta_data->>'full_name', ''),
        NEW.email
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger to create profile on user signup
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- Function to clean up expired sessions
CREATE OR REPLACE FUNCTION public.cleanup_expired_sessions()
RETURNS void AS $$
BEGIN
    -- Clean up expired liveness sessions
    UPDATE public.liveness_sessions 
    SET status = 'expired'
    WHERE expires_at < NOW() AND status IN ('pending', 'in_progress');
    
    -- Clean up expired auth sessions
    UPDATE public.auth_sessions 
    SET is_active = FALSE
    WHERE expires_at < NOW() AND is_active = TRUE;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- =====================================================================
-- STORAGE BUCKET
-- =====================================================================

-- Create storage bucket for face images (run this in Supabase Dashboard > Storage)
INSERT INTO storage.buckets (id, name, public) VALUES ('face-images', 'face-images', false);

-- Storage policies for face images bucket
CREATE POLICY "Users can upload their own face images" ON storage.objects
     FOR INSERT WITH CHECK (
         bucket_id = 'face-images' AND 
         auth.uid()::text = (storage.foldername(name))[1]
     );

 CREATE POLICY "Users can view their own face images" ON storage.objects
     FOR SELECT USING (
         bucket_id = 'face-images' AND 
         auth.uid()::text = (storage.foldername(name))[1]
     );

 CREATE POLICY "Users can delete their own face images" ON storage.objects
     FOR DELETE USING (
         bucket_id = 'face-images' AND 
         auth.uid()::text = (storage.foldername(name))[1]
     );

-- =====================================================================
-- SAMPLE DATA FOR TESTING (OPTIONAL)
-- =====================================================================

-- Uncomment to insert sample security events for testing
-- INSERT INTO public.security_events (event_type, severity, description, details) VALUES
-- ('system_startup', 'low', 'Face recognition system initialized', '{"version": "1.0", "models_loaded": true}'),
-- ('maintenance', 'medium', 'Database maintenance completed', '{"tables_optimized": 6, "indexes_rebuilt": 12}');

-- =====================================================================
-- COMPLETION MESSAGE
-- =====================================================================

DO $$
BEGIN
    RAISE NOTICE 'ZeroDay Face Recognition Database Schema created successfully!';
    RAISE NOTICE 'Tables created: profiles, face_embeddings, face_recognition_logs, liveness_sessions, auth_sessions, security_events';
    RAISE NOTICE 'Indexes, RLS policies, triggers, and functions configured.';
    RAISE NOTICE 'Next steps:';
    RAISE NOTICE '1. Create the face-images storage bucket in Supabase Dashboard';
    RAISE NOTICE '2. Configure storage policies (uncomment the storage policy section)';
    RAISE NOTICE '3. Test the setup with your application';
END $$;
