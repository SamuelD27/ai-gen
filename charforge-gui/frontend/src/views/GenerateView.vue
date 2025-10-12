<template>
  <AppLayout>
    <div class="max-w-7xl mx-auto space-y-6">
      <!-- Header -->
      <div class="flex items-center justify-between">
        <div>
          <h1 class="text-3xl font-bold bg-gradient-to-r from-primary via-purple-600 to-pink-600 bg-clip-text text-transparent">
            Image Generation
          </h1>
          <p class="text-muted-foreground mt-1">
            Create stunning images with state-of-the-art AI models
          </p>
        </div>

        <Button
          v-if="generatedImages.length > 0"
          @click="showHistory = true"
          variant="outline"
        >
          <History class="mr-2 h-4 w-4" />
          History ({{ generatedImages.length }})
        </Button>
      </div>

      <div class="grid gap-6 lg:grid-cols-3">
        <!-- Left Panel - Configuration -->
        <div class="lg:col-span-1 space-y-6">
          <!-- Model Selection -->
          <Card class="p-6">
            <h3 class="text-lg font-semibold mb-4 flex items-center">
              <Sparkles class="mr-2 h-5 w-5" />
              Model Selection
            </h3>

            <div class="space-y-4">
              <div v-if="loadingModels" class="text-center py-4">
                <div class="animate-spin rounded-full h-6 w-6 border-b-2 border-primary mx-auto"></div>
                <p class="text-sm text-muted-foreground mt-2">Loading models...</p>
              </div>

              <div v-else class="space-y-2">
                <div
                  v-for="model in availableModels"
                  :key="model.id"
                  @click="selectModel(model)"
                  :class="[
                    'p-4 rounded-lg border-2 cursor-pointer transition-all',
                    selectedModel?.id === model.id
                      ? 'border-primary bg-primary/5'
                      : 'border-border hover:border-primary/50'
                  ]"
                >
                  <div class="flex items-start justify-between">
                    <div class="flex-1">
                      <div class="flex items-center gap-2">
                        <p class="font-semibold">{{ model.name }}</p>
                        <span
                          v-if="model.recommended"
                          class="px-1.5 py-0.5 bg-primary/20 text-primary text-xs rounded"
                        >
                          Recommended
                        </span>
                      </div>
                      <p class="text-xs text-muted-foreground mt-1">{{ model.provider }}</p>
                    </div>
                    <Check v-if="selectedModel?.id === model.id" class="h-5 w-5 text-primary flex-shrink-0" />
                  </div>

                  <p class="text-sm text-muted-foreground mt-2">{{ model.description }}</p>

                  <div class="flex items-center gap-3 mt-3 text-xs">
                    <span class="flex items-center gap-1">
                      <Zap :class="['h-3 w-3', model.speed === 'fast' ? 'text-green-500' : model.speed === 'medium' ? 'text-yellow-500' : 'text-orange-500']" />
                      {{ model.speed }}
                    </span>
                    <span class="flex items-center gap-1">
                      <Award :class="['h-3 w-3', model.quality === 'excellent' ? 'text-primary' : 'text-muted-foreground']" />
                      {{ model.quality }}
                    </span>
                    <span class="text-muted-foreground">{{ model.vram_required }}</span>
                  </div>
                </div>
              </div>
            </div>
          </Card>

          <!-- Quick Presets -->
          <Card class="p-6">
            <h3 class="text-lg font-semibold mb-4">Quick Presets</h3>

            <div class="space-y-2">
              <Button
                v-for="preset in presets"
                :key="preset.id"
                @click="applyPreset(preset)"
                variant="outline"
                class="w-full justify-start text-left"
                size="sm"
              >
                <Wand2 class="mr-2 h-4 w-4" />
                <div class="flex-1">
                  <p class="font-medium">{{ preset.name }}</p>
                  <p class="text-xs text-muted-foreground">{{ preset.description }}</p>
                </div>
              </Button>
            </div>
          </Card>

          <!-- Memory Info -->
          <Card v-if="memoryInfo" class="p-6">
            <h3 class="text-sm font-semibold mb-2 flex items-center">
              <Cpu class="mr-2 h-4 w-4" />
              System Status
            </h3>
            <div class="space-y-1 text-sm">
              <div class="flex justify-between">
                <span class="text-muted-foreground">Device:</span>
                <span class="font-medium uppercase">{{ memoryInfo.device }}</span>
              </div>
              <div v-if="memoryInfo.device === 'cuda'" class="flex justify-between">
                <span class="text-muted-foreground">GPU Memory:</span>
                <span class="font-medium">{{ (memoryInfo.allocated_gb || 0).toFixed(2) }} GB</span>
              </div>
            </div>
          </Card>
        </div>

        <!-- Right Panel - Generation Interface -->
        <div class="lg:col-span-2 space-y-6">
          <!-- Prompt Input -->
          <Card class="p-6">
            <h3 class="text-lg font-semibold mb-4">Prompt</h3>

            <div class="space-y-4">
              <div>
                <textarea
                  v-model="form.prompt"
                  rows="4"
                  class="w-full px-4 py-3 border border-input rounded-lg bg-background resize-none focus:outline-none focus:ring-2 focus:ring-primary"
                  placeholder="Describe the image you want to create... Be detailed and specific!"
                  :disabled="isGenerating"
                ></textarea>
                <p class="text-xs text-muted-foreground mt-2">
                  Tip: Include details about style, lighting, composition, and quality
                </p>
              </div>

              <!-- Advanced Options Toggle -->
              <details class="group">
                <summary class="cursor-pointer text-sm font-medium flex items-center gap-2 hover:text-primary">
                  <ChevronRight class="h-4 w-4 transition-transform group-open:rotate-90" />
                  Advanced Options
                </summary>

                <div class="mt-4 space-y-4 pl-6">
                  <div>
                    <label class="block text-sm font-medium mb-2">Negative Prompt</label>
                    <textarea
                      v-model="form.negative_prompt"
                      rows="2"
                      class="w-full px-3 py-2 border border-input rounded-md bg-background resize-none text-sm"
                      placeholder="What to avoid in the image (e.g., blurry, distorted, low quality)"
                      :disabled="isGenerating"
                    ></textarea>
                  </div>

                  <div class="grid grid-cols-2 gap-4">
                    <div>
                      <label class="block text-sm font-medium mb-2">Width</label>
                      <Input
                        v-model.number="form.width"
                        type="number"
                        step="64"
                        min="256"
                        max="2048"
                        :disabled="isGenerating"
                      />
                    </div>
                    <div>
                      <label class="block text-sm font-medium mb-2">Height</label>
                      <Input
                        v-model.number="form.height"
                        type="number"
                        step="64"
                        min="256"
                        max="2048"
                        :disabled="isGenerating"
                      />
                    </div>
                  </div>

                  <div>
                    <label class="block text-sm font-medium mb-2">
                      Steps: {{ form.num_inference_steps }}
                    </label>
                    <input
                      v-model.number="form.num_inference_steps"
                      type="range"
                      min="10"
                      max="100"
                      step="5"
                      class="w-full"
                      :disabled="isGenerating"
                    />
                    <p class="text-xs text-muted-foreground mt-1">
                      More steps = better quality but slower
                    </p>
                  </div>

                  <div>
                    <label class="block text-sm font-medium mb-2">
                      Guidance Scale: {{ form.guidance_scale }}
                    </label>
                    <input
                      v-model.number="form.guidance_scale"
                      type="range"
                      min="1"
                      max="20"
                      step="0.5"
                      class="w-full"
                      :disabled="isGenerating"
                    />
                    <p class="text-xs text-muted-foreground mt-1">
                      How closely to follow the prompt
                    </p>
                  </div>

                  <div>
                    <label class="block text-sm font-medium mb-2">Number of Images</label>
                    <Input
                      v-model.number="form.num_images"
                      type="number"
                      min="1"
                      max="4"
                      :disabled="isGenerating"
                    />
                  </div>

                  <div>
                    <label class="block text-sm font-medium mb-2">Seed (Optional)</label>
                    <Input
                      v-model.number="form.seed"
                      type="number"
                      placeholder="Random if empty"
                      :disabled="isGenerating"
                    />
                    <p class="text-xs text-muted-foreground mt-1">
                      Use same seed for reproducible results
                    </p>
                  </div>
                </div>
              </details>
            </div>
          </Card>

          <!-- Generate Button -->
          <Button
            @click="generateImages"
            :disabled="!canGenerate || isGenerating"
            class="w-full"
            size="lg"
          >
            <Wand2 v-if="!isGenerating" class="mr-2 h-5 w-5" />
            <div v-else class="mr-2 animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
            {{ isGenerating ? 'Generating...' : 'Generate Images' }}
          </Button>

          <!-- Progress -->
          <div v-if="isGenerating" class="text-center space-y-2">
            <div class="h-2 bg-muted rounded-full overflow-hidden">
              <div class="h-full bg-gradient-to-r from-primary via-purple-600 to-pink-600 animate-pulse" style="width: 100%"></div>
            </div>
            <p class="text-sm text-muted-foreground">
              {{ generationProgress }}
            </p>
          </div>

          <!-- Generated Images -->
          <div v-if="currentGeneration && currentGeneration.images.length > 0">
            <Card class="p-6">
              <div class="flex items-center justify-between mb-4">
                <h3 class="text-lg font-semibold">Generated Images</h3>
                <div class="flex items-center gap-2 text-sm text-muted-foreground">
                  <Clock class="h-4 w-4" />
                  {{ currentGeneration.total_generation_time.toFixed(2) }}s
                </div>
              </div>

              <div :class="[
                'grid gap-4',
                currentGeneration.images.length === 1 ? 'grid-cols-1' : 'grid-cols-2'
              ]">
                <div
                  v-for="(image, index) in currentGeneration.images"
                  :key="index"
                  class="relative group"
                >
                  <div class="aspect-square bg-muted rounded-lg overflow-hidden">
                    <img
                      :src="`http://localhost:8000${image.url}`"
                      :alt="`Generated image ${index + 1}`"
                      class="w-full h-full object-contain"
                      @click="viewFullImage(image)"
                    />
                  </div>

                  <!-- Hover Overlay -->
                  <div class="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity rounded-lg flex items-center justify-center gap-2">
                    <Button
                      @click="viewFullImage(image)"
                      size="sm"
                      variant="secondary"
                    >
                      <Maximize2 class="h-4 w-4" />
                    </Button>
                    <Button
                      @click="downloadImage(image)"
                      size="sm"
                      variant="secondary"
                    >
                      <Download class="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </div>

              <!-- Generation Info -->
              <div class="mt-4 p-4 bg-muted rounded-lg text-sm space-y-2">
                <div class="flex items-start gap-2">
                  <span class="font-medium text-foreground">Prompt:</span>
                  <span class="text-muted-foreground flex-1">{{ currentGeneration.images[0].metadata.prompt }}</span>
                </div>
                <div class="flex items-center gap-4 text-xs text-muted-foreground">
                  <span>Model: {{ currentGeneration.model_info.name }}</span>
                  <span>•</span>
                  <span>{{ currentGeneration.images[0].metadata.width }}x{{ currentGeneration.images[0].metadata.height }}</span>
                  <span>•</span>
                  <span>{{ currentGeneration.images[0].metadata.steps }} steps</span>
                </div>
              </div>
            </Card>
          </div>

          <!-- Empty State -->
          <Card v-else-if="!isGenerating" class="p-12">
            <div class="text-center">
              <div class="mx-auto w-16 h-16 bg-gradient-to-br from-primary/20 via-purple-500/20 to-pink-500/20 rounded-full flex items-center justify-center mb-4">
                <Image class="h-8 w-8 text-primary" />
              </div>
              <h3 class="text-lg font-semibold mb-2">No images generated yet</h3>
              <p class="text-sm text-muted-foreground">
                Select a model, enter your prompt, and click "Generate Images" to begin
              </p>
            </div>
          </Card>
        </div>
      </div>
    </div>

    <!-- Full Image Modal -->
    <div
      v-if="fullImageView"
      class="fixed inset-0 bg-black/90 z-50 flex items-center justify-center p-4"
      @click="fullImageView = null"
    >
      <div class="relative max-w-6xl max-h-[90vh]" @click.stop>
        <Button
          @click="fullImageView = null"
          class="absolute -top-12 right-0"
          variant="ghost"
          size="sm"
        >
          <X class="h-6 w-6" />
        </Button>
        <img
          :src="`http://localhost:8000${fullImageView.url}`"
          :alt="fullImageView.metadata.prompt"
          class="max-w-full max-h-[90vh] object-contain rounded-lg"
        />
      </div>
    </div>
  </AppLayout>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useToast } from 'vue-toastification'
import axios from 'axios'
import {
  Sparkles, Wand2, Image, Check, Zap, Award, Cpu, ChevronRight,
  History, Clock, Maximize2, Download, X
} from 'lucide-vue-next'
import AppLayout from '@/components/layout/AppLayout.vue'
import Card from '@/components/ui/Card.vue'
import Button from '@/components/ui/Button.vue'
import Input from '@/components/ui/Input.vue'

const toast = useToast()

// State
const availableModels = ref<any[]>([])
const selectedModel = ref<any | null>(null)
const presets = ref<any[]>([])
const memoryInfo = ref<any>(null)
const loadingModels = ref(false)
const isGenerating = ref(false)
const generationProgress = ref('Initializing...')
const currentGeneration = ref<any>(null)
const generatedImages = ref<any[]>([])
const fullImageView = ref<any>(null)
const showHistory = ref(false)

const form = ref({
  prompt: '',
  negative_prompt: '',
  width: 1024,
  height: 1024,
  num_inference_steps: 30,
  guidance_scale: 7.5,
  num_images: 1,
  seed: null as number | null
})

// Computed
const canGenerate = computed(() => {
  return selectedModel.value && form.value.prompt.trim().length >= 3 && !isGenerating.value
})

// Methods
const loadModels = async () => {
  loadingModels.value = true
  try {
    const response = await axios.get('http://localhost:8000/api/generate/models')
    availableModels.value = response.data.models

    // Auto-select first recommended model
    const recommended = availableModels.value.find(m => m.recommended)
    if (recommended) {
      selectedModel.value = recommended
    }
  } catch (error: any) {
    toast.error('Failed to load models')
    console.error(error)
  } finally {
    loadingModels.value = false
  }
}

const loadPresets = async () => {
  try {
    const response = await axios.get('http://localhost:8000/api/generate/presets')
    presets.value = response.data
  } catch (error: any) {
    console.error('Failed to load presets:', error)
  }
}

const loadMemoryInfo = async () => {
  try {
    const response = await axios.get('http://localhost:8000/api/generate/memory')
    memoryInfo.value = response.data
  } catch (error: any) {
    console.error('Failed to load memory info:', error)
  }
}

const selectModel = (model: any) => {
  selectedModel.value = model
  toast.info(`Selected ${model.name}`)
}

const applyPreset = (preset: any) => {
  // Find and select the preset's model
  const model = availableModels.value.find(m => m.id === preset.model_id)
  if (model) {
    selectedModel.value = model
  }

  // Apply preset values
  form.value.width = preset.width
  form.value.height = preset.height
  form.value.num_inference_steps = preset.steps
  form.value.guidance_scale = preset.guidance_scale

  // Optionally update prompt with example
  if (!form.value.prompt) {
    form.value.prompt = preset.example_prompt
  }

  toast.info(`Applied ${preset.name} preset`)
}

const generateImages = async () => {
  if (!canGenerate.value) return

  isGenerating.value = true
  generationProgress.value = 'Loading model...'

  try {
    const requestData = {
      prompt: form.value.prompt,
      negative_prompt: form.value.negative_prompt || undefined,
      model_id: selectedModel.value.id,
      width: form.value.width,
      height: form.value.height,
      num_inference_steps: form.value.num_inference_steps,
      guidance_scale: form.value.guidance_scale,
      num_images: form.value.num_images,
      seed: form.value.seed || undefined
    }

    generationProgress.value = 'Generating images...'

    const response = await axios.post('http://localhost:8000/api/generate/generate', requestData)

    currentGeneration.value = response.data
    generatedImages.value.unshift(response.data)

    toast.success(`Generated ${response.data.images.length} image(s) successfully!`)

    // Refresh memory info
    await loadMemoryInfo()

  } catch (error: any) {
    const errorMsg = error.response?.data?.detail || 'Failed to generate images'
    toast.error(errorMsg)
    console.error(error)
  } finally {
    isGenerating.value = false
    generationProgress.value = ''
  }
}

const viewFullImage = (image: any) => {
  fullImageView.value = image
}

const downloadImage = (image: any) => {
  const link = document.createElement('a')
  link.href = `http://localhost:8000${image.url}`
  link.download = image.filename
  link.click()
  toast.success('Download started')
}

// Lifecycle
onMounted(() => {
  loadModels()
  loadPresets()
  loadMemoryInfo()
})
</script>
