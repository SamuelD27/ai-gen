<template>
  <input
    :value="modelValue"
    @input="handleInput"
    :class="cn(
      'flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50',
      $attrs.class ?? ''
    )"
    v-bind="omit($attrs, ['class'])"
  />
</template>

<script setup lang="ts">
import { cn } from '@/lib/utils'

interface Props {
  modelValue?: string | number
}

defineProps<Props>()

const emit = defineEmits<{
  (e: 'update:modelValue', value: string): void
}>()

// Helper to omit specific attributes
const omit = (obj: Record<string, any>, keys: string[]) => {
  const result = { ...obj }
  keys.forEach(key => delete result[key])
  return result
}

// Debug: log when input changes
const handleInput = (event: Event) => {
  const value = (event.target as HTMLInputElement).value
  console.log('📝 Input changed:', value)
  emit('update:modelValue', value)
}
</script>
