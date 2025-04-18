<script setup lang="ts">
import { ref, onMounted } from 'vue';
import videoApi from '../api/api';
import type { VideoListItem } from '../types/types';

const videos = ref<VideoListItem[]>([]);
const loading = ref(true);
const error = ref<string | null>(null);

const fetchVideos = async () => {
  try {
    loading.value = true;
    const response = await videoApi.listVideos();
    console.log(response);
    videos.value = response;
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'Failed to fetch videos';
  } finally {
    loading.value = false;
  }
};

onMounted(() => {
  fetchVideos();
});
</script>

<template>
  <main class="w-screen min-h-screen text-white p-6">
    <div class="mx-auto">
      <h1 class="text-3xl font-bold mb-8">Available Videos</h1>

      <!-- Loading State -->
      <div v-if="loading" class="flex justify-center items-center h-64">
        <div class="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-white"></div>
      </div>

      <!-- Error State -->
      <div v-else-if="error" class="bg-red-500/10 border border-red-500 rounded-lg p-4 text-red-500">
        {{ error }}
      </div>

      <!-- Videos Grid -->
      <div v-else class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div 
          v-for="video in videos" 
          :key="video.name"
          class="bg-gray-800 rounded-lg overflow-hidden hover:bg-gray-700 transition-colors shadow-lg"
        >
          <div class="p-4">
            <h3 class="text-xl font-semibold mb-2">{{ video.name }}</h3>
            <div class="text-gray-400 text-sm">
              Last Analyzed: {{ new Date(video.last_analyzed).toLocaleString() }}
            </div>
            <div class="mt-4 flex justify-end">
              <router-link 
                :to="{ name: 'video', params: { name: video.name }}" 
                class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors"
              >
                View Details
              </router-link>
            </div>
          </div>
        </div>
      </div>

      <!-- Empty State -->
      <div v-if="!loading && !error && videos && videos.length === 0" class="text-center py-12">
        <h3 class="text-xl text-gray-400">No videos available</h3>
      </div>
    </div>
  </main>
</template> 