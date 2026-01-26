"use client";

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { supabase } from '@/lib/supabase';
import { endpoints } from '@/lib/api';
import { handleError } from '@/lib/errorHandling';
import { useStorageSync, broadcastStorageChange } from '@/hooks/useStorageSync';
import type { Garment, TryOnResult, Recommendation } from '@/lib/types';

interface PersonalImage {
  url: string;
  type: 'head-only' | 'full-body';
  uploadedAt: Date;
}

interface StudioContextState {
  // Data
  personalImage: PersonalImage | null;
  garments: Garment[];
  tryOnResults: TryOnResult[];
  recommendations: Recommendation[];
  selectedGarment: Garment | null;
  
  // Loading states
  isLoadingPersonalImage: boolean;
  isLoadingGarments: boolean;
  isGenerating: boolean;
  isLoadingRecommendations: boolean;
  
  // UI state
  hasRequestedRecommendations: boolean;
  
  // User
  userId: string | null;
}

interface StudioContextActions {
  // Garment actions
  addGarment: (garment: Garment) => void;
  removeGarment: (garmentId: string) => void;
  selectGarment: (garment: Garment | null) => void;
  refreshGarments: () => Promise<void>;
  
  // Try-on actions
  addTryOnResult: (result: TryOnResult) => void;
  setIsGenerating: (isGenerating: boolean) => void;
  
  // Recommendation actions
  setRecommendations: (recommendations: Recommendation[]) => void;
  setIsLoadingRecommendations: (isLoading: boolean) => void;
  setHasRequestedRecommendations: (hasRequested: boolean) => void;
  
  // Personal image actions
  setPersonalImage: (image: PersonalImage | null) => void;
  
  // State persistence
  saveState: () => Promise<void>;
  loadState: () => Promise<void>;
}

type StudioContextType = StudioContextState & StudioContextActions;

const StudioContext = createContext<StudioContextType | undefined>(undefined);

const STORAGE_KEY = 'studio_state';
const STATE_VERSION = 1;

interface PersistedState {
  version: number;
  selectedGarmentId: string | null;
  hasRequestedRecommendations: boolean;
  tryOnResults: TryOnResult[];
  recommendations: Recommendation[];
  timestamp: number;
}

export function StudioProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<StudioContextState>({
    personalImage: null,
    garments: [],
    tryOnResults: [],
    recommendations: [],
    selectedGarment: null,
    isLoadingPersonalImage: false,
    isLoadingGarments: false,
    isGenerating: false,
    isLoadingRecommendations: false,
    hasRequestedRecommendations: false,
    userId: null,
  });

  // Initialize user session and load data
  useEffect(() => {
    const initializeStudio = async () => {
      try {
        const { data: { session } } = await supabase.auth.getSession();
        
        if (!session) {
          return;
        }

        setState(prev => ({ ...prev, userId: session.user.id }));

        // Load persisted state first
        await loadPersistedState(session.user.id);

        // Load data from Supabase
        setState(prev => ({ 
          ...prev, 
          isLoadingPersonalImage: true, 
          isLoadingGarments: true 
        }));

        const [personalImageResult, garmentsResult] = await Promise.allSettled([
          endpoints.getPersonalImage(session.user.id),
          endpoints.listGarments(session.user.id),
        ]);

        setState(prev => ({
          ...prev,
          personalImage: personalImageResult.status === 'fulfilled' ? personalImageResult.value : null,
          garments: garmentsResult.status === 'fulfilled' ? garmentsResult.value : [],
          isLoadingPersonalImage: false,
          isLoadingGarments: false,
        }));
      } catch (error) {
        const session = await supabase.auth.getSession().then(r => r.data.session);
        handleError(error, 'initialize studio', { 
          showToast: true,
          userId: session?.user.id 
        });
        setState(prev => ({ 
          ...prev, 
          isLoadingPersonalImage: false, 
          isLoadingGarments: false 
        }));
      }
    };

    initializeStudio();
  }, []);

  // Auto-save state when it changes
  useEffect(() => {
    if (state.userId) {
      const timeoutId = setTimeout(() => {
        savePersistedState();
      }, 500); // Reduced debounce for faster saves

      return () => clearTimeout(timeoutId);
    }
  }, [state.selectedGarment, state.hasRequestedRecommendations, state.userId, state.tryOnResults, state.recommendations]);

  // Cross-tab synchronization for garment changes
  const GARMENT_SYNC_KEY = 'studio_garments_sync';
  
  useStorageSync(GARMENT_SYNC_KEY, useCallback((newValue) => {
    if (newValue && state.userId) {
      try {
        const syncData = JSON.parse(newValue);
        if (syncData.userId === state.userId && syncData.timestamp > Date.now() - 5000) {
          // Refresh garments from server to stay in sync
          refreshGarments();
        }
      } catch (error) {
        console.error('Failed to parse garment sync data:', error);
      }
    }
  }, [state.userId]));

  // Broadcast garment changes to other tabs
  const broadcastGarmentChange = useCallback(() => {
    if (state.userId) {
      broadcastStorageChange(GARMENT_SYNC_KEY, JSON.stringify({
        userId: state.userId,
        timestamp: Date.now(),
      }));
    }
  }, [state.userId]);

  // Load persisted state from localStorage
  const loadPersistedState = async (userId: string) => {
    try {
      const stored = localStorage.getItem(`${STORAGE_KEY}_${userId}`);
      if (!stored) return;

      const parsed: PersistedState = JSON.parse(stored);
      
      // Check version compatibility
      if (parsed.version !== STATE_VERSION) {
        localStorage.removeItem(`${STORAGE_KEY}_${userId}`);
        return;
      }

      // Check if state is not too old (24 hours)
      const age = Date.now() - parsed.timestamp;
      if (age > 24 * 60 * 60 * 1000) {
        localStorage.removeItem(`${STORAGE_KEY}_${userId}`);
        return;
      }

      setState(prev => ({
        ...prev,
        hasRequestedRecommendations: parsed.hasRequestedRecommendations,
        tryOnResults: parsed.tryOnResults || [],
        recommendations: parsed.recommendations || [],
      }));

      // Selected garment will be restored after garments are loaded
      if (parsed.selectedGarmentId) {
        // Store for later restoration
        sessionStorage.setItem('pendingSelectedGarmentId', parsed.selectedGarmentId);
      }
    } catch (error) {
      console.error('Failed to load persisted state:', error);
    }
  };

  // Save state to localStorage
  const savePersistedState = async () => {
    if (!state.userId) return;

    try {
      const toSave: PersistedState = {
        version: STATE_VERSION,
        selectedGarmentId: state.selectedGarment?.id || null,
        hasRequestedRecommendations: state.hasRequestedRecommendations,
        tryOnResults: state.tryOnResults,
        recommendations: state.recommendations,
        timestamp: Date.now(),
      };

      localStorage.setItem(`${STORAGE_KEY}_${state.userId}`, JSON.stringify(toSave));
    } catch (error) {
      console.error('Failed to save persisted state:', error);
    }
  };

  // Restore selected garment after garments are loaded
  useEffect(() => {
    const pendingId = sessionStorage.getItem('pendingSelectedGarmentId');
    if (pendingId && state.garments.length > 0 && !state.selectedGarment) {
      const garment = state.garments.find(g => g.id === pendingId);
      if (garment) {
        setState(prev => ({ ...prev, selectedGarment: garment }));
      }
      sessionStorage.removeItem('pendingSelectedGarmentId');
    }
  }, [state.garments, state.selectedGarment]);

  // Actions
  const addGarment = useCallback((garment: Garment) => {
    setState(prev => ({
      ...prev,
      garments: [garment, ...prev.garments],
    }));
    broadcastGarmentChange();
  }, [broadcastGarmentChange]);

  const removeGarment = useCallback((garmentId: string) => {
    setState(prev => ({
      ...prev,
      garments: prev.garments.filter(g => g.id !== garmentId),
      selectedGarment: prev.selectedGarment?.id === garmentId ? null : prev.selectedGarment,
    }));
    broadcastGarmentChange();
  }, [broadcastGarmentChange]);

  const selectGarment = useCallback((garment: Garment | null) => {
    setState(prev => ({ ...prev, selectedGarment: garment }));
  }, []);

  const refreshGarments = useCallback(async () => {
    if (!state.userId) return;

    try {
      setState(prev => ({ ...prev, isLoadingGarments: true }));
      const garments = await endpoints.listGarments(state.userId);
      setState(prev => ({ ...prev, garments, isLoadingGarments: false }));
    } catch (error) {
      handleError(error, 'refresh garments', { 
        showToast: false, // Don't show toast for background refresh
        userId: state.userId 
      });
      setState(prev => ({ ...prev, isLoadingGarments: false }));
      throw error;
    }
  }, [state.userId]);

  const addTryOnResult = useCallback((result: TryOnResult) => {
    setState(prev => ({
      ...prev,
      tryOnResults: [result, ...prev.tryOnResults],
    }));
  }, []);

  const setIsGenerating = useCallback((isGenerating: boolean) => {
    setState(prev => ({ ...prev, isGenerating }));
  }, []);

  const setRecommendations = useCallback((recommendations: Recommendation[]) => {
    setState(prev => ({ ...prev, recommendations }));
  }, []);

  const setIsLoadingRecommendations = useCallback((isLoading: boolean) => {
    setState(prev => ({ ...prev, isLoadingRecommendations: isLoading }));
  }, []);

  const setHasRequestedRecommendations = useCallback((hasRequested: boolean) => {
    setState(prev => ({ ...prev, hasRequestedRecommendations: hasRequested }));
  }, []);

  const setPersonalImage = useCallback((image: PersonalImage | null) => {
    setState(prev => ({ ...prev, personalImage: image }));
  }, []);

  const saveState = useCallback(async () => {
    await savePersistedState();
  }, [state.userId, state.selectedGarment, state.hasRequestedRecommendations]);

  const loadState = useCallback(async () => {
    if (state.userId) {
      await loadPersistedState(state.userId);
    }
  }, [state.userId]);

  const contextValue: StudioContextType = {
    ...state,
    addGarment,
    removeGarment,
    selectGarment,
    refreshGarments,
    addTryOnResult,
    setIsGenerating,
    setRecommendations,
    setIsLoadingRecommendations,
    setHasRequestedRecommendations,
    setPersonalImage,
    saveState,
    loadState,
  };

  return (
    <StudioContext.Provider value={contextValue}>
      {children}
    </StudioContext.Provider>
  );
}

export function useStudio() {
  const context = useContext(StudioContext);
  if (context === undefined) {
    throw new Error('useStudio must be used within a StudioProvider');
  }
  return context;
}
