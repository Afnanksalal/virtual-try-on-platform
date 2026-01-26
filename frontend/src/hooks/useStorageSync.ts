import { useEffect } from 'react';

/**
 * Hook to synchronize state across browser tabs/windows using storage events
 */
export function useStorageSync(key: string, callback: (newValue: string | null) => void) {
  useEffect(() => {
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === key && e.newValue !== e.oldValue) {
        callback(e.newValue);
      }
    };

    // Listen for storage events from other tabs
    window.addEventListener('storage', handleStorageChange);

    return () => {
      window.removeEventListener('storage', handleStorageChange);
    };
  }, [key, callback]);
}

/**
 * Broadcast a change to other tabs
 */
export function broadcastStorageChange(key: string, value: string) {
  localStorage.setItem(key, value);
  
  // Dispatch custom event for same-tab synchronization
  window.dispatchEvent(new CustomEvent('local-storage-change', {
    detail: { key, value }
  }));
}

/**
 * Hook to listen for same-tab storage changes
 */
export function useLocalStorageSync(key: string, callback: (newValue: string) => void) {
  useEffect(() => {
    const handleLocalChange = (e: Event) => {
      const customEvent = e as CustomEvent<{ key: string; value: string }>;
      if (customEvent.detail.key === key) {
        callback(customEvent.detail.value);
      }
    };

    window.addEventListener('local-storage-change', handleLocalChange);

    return () => {
      window.removeEventListener('local-storage-change', handleLocalChange);
    };
  }, [key, callback]);
}
