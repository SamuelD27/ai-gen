<template>
  <AppLayout>
    <div class="max-w-6xl mx-auto space-y-6">
      <!-- Header -->
      <div class="flex items-center justify-between">
        <div>
          <h1 class="text-3xl font-bold bg-gradient-to-r from-primary via-purple-600 to-pink-600 bg-clip-text text-transparent">
            Video Generation
          </h1>
          <p class="text-muted-foreground mt-1">
            Create professional AI-generated videos with Sora 2, VEO3, and Runway
          </p>
        </div>
      </div>

      <!-- Generation Form -->
      <div class="grid lg:grid-cols-3 gap-6">
        <!-- Left Panel - Settings -->
        <div class="lg:col-span-1 space-y-6">
          <Card>
            <div class="p-6 space-y-4">
              <h3 class="font-semibold text-lg">Settings</h3>

              <!-- Provider Selection -->
              <div class="space-y-2">
                <label class="text-sm font-medium">Provider</label>
                <select
                  v-model="selectedProvider"
                  class="w-full px-3 py-2 border border-border rounded-md bg-background"
                >
                  <option
                    v-for="provider in availableProviders"
                    :key="provider.id"
                    :value="provider.id"
                    :disabled="!provider.available"
                  >
                    {{ provider.name }}
                    {{ !provider.available ? '(Not configured)' : '' }}
                  </option>
                </select>
                <p v-if="currentProvider" class="text-xs text-muted-foreground">
                  {{ currentProvider.description }}
                </p>
              </div>

              <!-- Duration -->
              <div class="space-y-2">
                <label class="text-sm font-medium">
                  Duration: {{ duration }}s
                </label>
                <input
                  v-model.number="duration"
                  type="range"
                  min="3"
                  :max="currentProvider?.max_duration || 60"
                  step="1"
                  class="w-full"
                />
                <div class="flex justify-between text-xs text-muted-foreground">
                  <span>3s</span>
                  <span>{{ currentProvider?.max_duration || 60 }}s</span>
                </div>
              </div>

              <!-- Resolution -->
              <div class="space-y-2">
                <label class="text-sm font-medium">Resolution</label>
                <select
                  v-model="resolution"
                  class="w-full px-3 py-2 border border-border rounded-md bg-background"
                >
                  <option
                    v-for="res in currentProvider?.resolutions || ['1080p']"
                    :key="res"
                    :value="res"
                  >
                    {{ res }}
                  </option>
                </select>
              </div>

              <!-- Motion Intensity -->
              <div class="space-y-2">
                <label class="text-sm font-medium">
                  Motion Intensity: {{ (motionIntensity * 100).toFixed(0) }}%
                </label>
                <input
                  v-model.number="motionIntensity"
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  class="w-full"
                />
                <div class="flex justify-between text-xs text-muted-foreground">
                  <span>Static</span>
                  <span>Dynamic</span>
                </div>
              </div>

              <!-- Advanced Settings -->
              <details class="space-y-2">
                <summary class="text-sm font-medium cursor-pointer">Advanced Settings</summary>
                <div class="space-y-3 mt-3">
                  <div class="space-y-2">
                    <label class="text-sm font-medium">Aspect Ratio</label>
                    <select
                      v-model="aspectRatio"
                      class="w-full px-3 py-2 border border-border rounded-md bg-background text-sm"
                    >
                      <option value="16:9">16:9 (Widescreen)</option>
                      <option value="9:16">9:16 (Vertical)</option>
                      <option value="1:1">1:1 (Square)</option>
                      <option value="4:3">4:3 (Standard)</option>
                    </select>
                  </div>

                  <div class="space-y-2">
                    <label class="text-sm font-medium">FPS</label>
                    <select
                      v-model.number="fps"
                      class="w-full px-3 py-2 border border-border rounded-md bg-background text-sm"
                    >
                      <option :value="24">24 FPS (Cinematic)</option>
                      <option :value="30">30 FPS (Standard)</option>
                      <option :value="60">60 FPS (Smooth)</option>
                    </select>
                  </div>

                  <div class="space-y-2">
                    <label class="text-sm font-medium">Seed (Optional)</label>
                    <input
                      v-model.number="seed"
                      type="number"
                      placeholder="Random"
                      class="w-full px-3 py-2 border border-border rounded-md bg-background text-sm"
                    />
                  </div>
                </div>
              </details>
            </div>
          </Card>

          <!-- Provider Info -->
          <Card v-if="currentProvider">
            <div class="p-4 space-y-2">
              <h4 class="font-medium text-sm">Provider Info</h4>
              <div class="grid grid-cols-2 gap-2 text-xs">
                <div class="flex items-center space-x-2">
                  <Zap class="h-3 w-3 text-primary" />
                  <span class="text-muted-foreground">Speed:</span>
                  <span class="font-medium">{{ currentProvider.speed }}</span>
                </div>
                <div class="flex items-center space-x-2">
                  <Sparkles class="h-3 w-3 text-primary" />
                  <span class="text-muted-foreground">Quality:</span>
                  <span class="font-medium">{{ currentProvider.quality }}</span>
                </div>
              </div>
            </div>
          </Card>
        </div>

        <!-- Right Panel - Prompt & Preview -->
        <div class="lg:col-span-2 space-y-6">
          <!-- Prompt Input -->
          <Card>
            <div class="p-6 space-y-4">
              <h3 class="font-semibold text-lg">Describe Your Video</h3>
              <textarea
                v-model="prompt"
                placeholder="A serene lake at sunset with mountains in the background, cinematic camera movement..."
                class="w-full h-32 px-3 py-2 border border-border rounded-md bg-background resize-none"
                :disabled="isGenerating"
              />
              <div class="flex items-center justify-between">
                <p class="text-xs text-muted-foreground">
                  {{ prompt.length }}/1000 characters
                </p>
                <button
                  @click="generateVideo"
                  :disabled="!canGenerate"
                  class="px-6 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
                >
                  <Film class="h-4 w-4" />
                  <span>{{ isGenerating ? 'Generating...' : 'Generate Video' }}</span>
                </button>
              </div>
            </div>
          </Card>

          <!-- Video Preview -->
          <Card v-if="generatedVideo || isGenerating">
            <div class="p-6 space-y-4">
              <h3 class="font-semibold text-lg">Generated Video</h3>

              <!-- Loading State -->
              <div v-if="isGenerating" class="space-y-4">
                <div class="aspect-video bg-muted rounded-lg flex items-center justify-center">
                  <div class="text-center space-y-3">
                    <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
                    <p class="text-sm text-muted-foreground">
                      Generating your video...
                    </p>
                    <p class="text-xs text-muted-foreground">
                      This may take 1-3 minutes
                    </p>
                  </div>
                </div>
              </div>

              <!-- Video Player -->
              <div v-else-if="generatedVideo" class="space-y-4">
                <div class="aspect-video bg-black rounded-lg overflow-hidden">
                  <video
                    :src="generatedVideo.video_url"
                    controls
                    class="w-full h-full"
                  />
                </div>

                <!-- Video Info -->
                <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <p class="text-muted-foreground">Duration</p>
                    <p class="font-medium">{{ generatedVideo.duration }}s</p>
                  </div>
                  <div>
                    <p class="text-muted-foreground">Resolution</p>
                    <p class="font-medium">{{ generatedVideo.resolution }}</p>
                  </div>
                  <div>
                    <p class="text-muted-foreground">Provider</p>
                    <p class="font-medium">{{ generatedVideo.provider }}</p>
                  </div>
                  <div>
                    <p class="text-muted-foreground">Generation Time</p>
                    <p class="font-medium">{{ generatedVideo.generation_time.toFixed(1) }}s</p>
                  </div>
                </div>

                <!-- Actions -->
                <div class="flex space-x-2">
                  <a
                    :href="generatedVideo.video_url"
                    download
                    class="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 flex items-center space-x-2"
                  >
                    <Download class="h-4 w-4" />
                    <span>Download</span>
                  </a>
                  <button
                    @click="generatedVideo = null"
                    class="px-4 py-2 border border-border rounded-md hover:bg-accent"
                  >
                    Generate New
                  </button>
                </div>
              </div>
            </div>
          </Card>

          <!-- Example Prompts -->
          <Card v-if="!generatedVideo && !isGenerating">
            <div class="p-6 space-y-3">
              <h3 class="font-semibold">Example Prompts</h3>
              <div class="grid gap-2">
                <button
                  v-for="example in examplePrompts"
                  :key="example"
                  @click="prompt = example"
                  class="text-left px-3 py-2 rounded-md border border-border hover:bg-accent text-sm"
                >
                  {{ example }}
                </button>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  </AppLayout>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useToast } from 'vue-toastification'
import { Film, Zap, Sparkles, Download } from 'lucide-vue-next'
import AppLayout from '@/components/layout/AppLayout.vue'
import Card from '@/components/ui/Card.vue'
import axios from 'axios'

const toast = useToast()

// State
const prompt = ref('')
const selectedProvider = ref('veo3-fast')
const duration = ref(5)
const resolution = ref('1080p')
const motionIntensity = ref(0.7)
const aspectRatio = ref('16:9')
const fps = ref(24)
const seed = ref<number | null>(null)
const isGenerating = ref(false)
const generatedVideo = ref<any>(null)
const availableProviders = ref<any[]>([])

// Example prompts
const examplePrompts = [
  'A serene lake at sunset with mountains in the background, cinematic camera movement',
  'Time-lapse of a bustling city street transitioning from day to night',
  'Close-up of a blooming flower opening in fast motion, macro photography style',
  'Aerial drone shot flying over a tropical beach with crystal clear water',
  'Cozy coffee shop interior with morning light streaming through windows',
]

// Computed
const currentProvider = computed(() => {
  return availableProviders.value.find(p => p.id === selectedProvider.value)
})

const canGenerate = computed(() => {
  return prompt.value.length >= 10 && !isGenerating.value && currentProvider.value?.available
})

// Methods
const fetchProviders = async () => {
  try {
    const response = await axios.get('/api/video/providers')
    availableProviders.value = response.data

    // Select first available provider
    const firstAvailable = availableProviders.value.find(p => p.available)
    if (firstAvailable) {
      selectedProvider.value = firstAvailable.id
    }
  } catch (error: any) {
    console.error('Failed to fetch providers:', error)
    toast.error('Failed to load video providers')
  }
}

const generateVideo = async () => {
  if (!canGenerate.value) return

  isGenerating.value = true
  generatedVideo.value = null

  try {
    const response = await axios.post('/api/video/generate', {
      prompt: prompt.value,
      provider: selectedProvider.value,
      duration: duration.value,
      resolution: resolution.value,
      motion_intensity: motionIntensity.value,
      aspect_ratio: aspectRatio.value,
      fps: fps.value,
      seed: seed.value,
    })

    generatedVideo.value = response.data
    toast.success('Video generated successfully!')
  } catch (error: any) {
    console.error('Video generation failed:', error)
    const errorMsg = error.response?.data?.detail || 'Failed to generate video'
    toast.error(errorMsg)
  } finally {
    isGenerating.value = false
  }
}

// Lifecycle
onMounted(() => {
  fetchProviders()
})
</script>
