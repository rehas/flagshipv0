import { createRouter, createWebHistory } from 'vue-router'
import VideoListView from '../views/VideoListView.vue'
import VideoDetailView from '../views/VideoDetailView.vue'

const router = createRouter({
  history: createWebHistory('/'),
  routes: [
    {
      path: '/',
      name: 'videos',
      component: VideoListView
    },
    {
      path: '/video/:name',
      name: 'video',
      component: VideoDetailView
    }
  ]
})

export default router
