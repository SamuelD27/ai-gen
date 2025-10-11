import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export interface User {
  id: number
  username: string
  email: string
  is_active: boolean
}

// Simplified auth store - no authentication required
export const useAuthStore = defineStore('auth', () => {
  // Always return a default user (no auth)
  const user = ref<User>({
    id: 1,
    username: 'default_user',
    email: 'user@charforge.local',
    is_active: true
  })

  const isLoading = ref(false)
  const authEnabled = ref(false) // Always disabled

  // Always authenticated
  const isAuthenticated = computed(() => true)

  // No-op functions for compatibility
  const initializeAuth = async () => {
    // Do nothing - no auth needed
  }

  const checkAuthEnabled = async () => {
    // Do nothing - auth is always disabled
  }

  return {
    // State
    user,
    isLoading,
    authEnabled,

    // Getters
    isAuthenticated,

    // Actions (no-ops for compatibility)
    initializeAuth,
    checkAuthEnabled
  }
})
