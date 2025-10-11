<template>
  <div id="app" class="min-h-screen bg-background">
    <router-view />

    <!-- Onboarding Components -->
    <WelcomeModal />
    <OnboardingTour />
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue'
import { useOnboardingStore } from '@/stores/onboarding'
import WelcomeModal from '@/components/onboarding/WelcomeModal.vue'
import OnboardingTour from '@/components/onboarding/OnboardingTour.vue'

const onboardingStore = useOnboardingStore()

onMounted(() => {
  // Initialize onboarding state
  onboardingStore.initializeOnboarding()
})
</script>

<style scoped>
#app {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
</style>
