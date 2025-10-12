import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      redirect: '/dashboard'
    },
    {
      path: '/dashboard',
      name: 'Dashboard',
      component: () => import('@/views/DashboardView.vue')
    },
    {
      path: '/characters',
      name: 'Characters',
      component: () => import('@/views/CharactersView.vue')
    },
    {
      path: '/characters/create',
      name: 'CreateCharacter',
      component: () => import('@/views/CreateCharacterView.vue')
    },
    {
      path: '/characters/:id',
      name: 'CharacterDetail',
      component: () => import('@/views/CharacterDetailView.vue')
    },
    {
      path: '/training',
      name: 'Training',
      component: () => import('@/views/TrainingView.vue')
    },
    {
      path: '/inference',
      name: 'Inference',
      component: () => import('@/views/InferenceView.vue')
    },
    {
      path: '/gallery',
      name: 'Gallery',
      component: () => import('@/views/GalleryView.vue')
    },
    {
      path: '/video',
      name: 'VideoGeneration',
      component: () => import('@/views/VideoGenerationView.vue')
    },
    {
      path: '/media',
      name: 'Media',
      component: () => import('@/views/MediaView.vue')
    },
    {
      path: '/datasets',
      name: 'Datasets',
      component: () => import('@/views/DatasetsView.vue')
    },
    {
      path: '/settings',
      name: 'Settings',
      component: () => import('@/views/SettingsView.vue')
    },
    {
      path: '/:pathMatch(.*)*',
      name: 'NotFound',
      component: () => import('@/views/NotFoundView.vue')
    }
  ]
})

// No authentication required - all routes are accessible
export default router
