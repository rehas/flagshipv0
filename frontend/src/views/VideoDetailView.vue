<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { useRoute } from 'vue-router';
import videoApi from '../api/api';
import type { VideoStats, VideoAnalytics, HeatmapData, Video } from '../types/types';
import config from '../config';
const route = useRoute();
const videoName = route.params.name as string;

const videoStats = ref<VideoStats>({
  viewers: 0,
  likes: 0,
  comments: 0,
  duration: '00:00:00',
  quality: 'HD'
});

const analytics = ref<VideoAnalytics | null>(null);
const heatmapData = ref<HeatmapData[]>([]);
const videoUrl = ref<string>(''); 
const heatmap =  ref<string>(''); 
const videoElement = ref<HTMLVideoElement | null>(null);
const loading = ref(true);
const error = ref<string | null>(null);
const backendUrl = config.apiUrl;
onMounted(async () => {
  try {
    loading.value = true;
    const analyticsResponse = await videoApi.getAnalytics(videoName);
    analytics.value = analyticsResponse;
    videoUrl.value = `${backendUrl}/stream/${videoName}`;
    heatmap.value = `${backendUrl}/${analytics.value.heatmap_image_path}`

    
    if (videoElement.value) {
      videoElement.value.autoplay = true;
      videoElement.value.muted = true;
    }
    

    // Set up periodic updates
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'Failed to load video';
  } finally {
    loading.value = false;
  }
});


const getMostDistanceTraveled = (distanceTraveled: object) => {
  const distances = Object.values(distanceTraveled);
  return Math.max(...distances);
  };

const getEmployeeWithMostDistanceTraveled = (distanceTraveled: object) => {
  const distances = Object.values(distanceTraveled);
  const maxDistance = Math.max(...distances);
  const employee = Object.keys(distanceTraveled).find(key => distanceTraveled[key] === maxDistance);
  return employee || 'N/A';
};

const getTotalDistanceTraveled = (distanceTraveled: object) => {
  const distances = Object.values(distanceTraveled);
  return parseFloat(distances.reduce((sum, distance) => sum + distance, 0).toFixed(2));
};
</script>

<template>
  <main class="w-screen h-screen flex flex-col lg:flex-row">
    <!-- Loading State -->
    <div v-if="loading" class="w-full h-full flex items-center justify-center bg-gray-900">
      <div class="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-white"></div>
    </div>

    <!-- Error State -->
    <div v-else-if="error" class="w-full h-full flex items-center justify-center bg-gray-900">
      <div class="bg-red-500/10 border border-red-500 rounded-lg p-4 text-red-500">
        {{ error }}
      </div>
    </div>

    <!-- Content -->
    <template v-else>
      <!-- Video Section -->
      <div class="w-full lg:w-2/3 bg-gray-900">
        <video
          ref="videoElement"
          class="w-full h-full object-cover"
          controls
          playsinline
          autoplay
        >
          <source :src="videoUrl" type="video/mp4">
          Your browser does not support the video tag.
        </video>
      </div>
      
      <!-- Stats Section -->
      <div class="w-full lg:w-1/3 p-4 overflow-y-auto mb-6 gap-2">
        <h2 class="text-xl font-bold mb-4 text-white">Stats</h2>
        <div class="grid grid-cols-2 gap-3">
          <div class="bg-gray-700 p-3 rounded-lg flex flex-col items-center">
            <span class="text-gray-400 text-sm">Total Employees</span>
            <span class="text-lg font-semibold text-white">{{ analytics.employee_count }}</span>
          </div>
          <div class="bg-gray-700 p-3 rounded-lg flex flex-col items-center">
            <span class="text-gray-400 text-sm">Total Distance Traveled</span>
            <span class="text-lg font-semibold text-white">{{ getTotalDistanceTraveled(analytics.distance_traveled) }}</span>
          </div>
          <div class="bg-gray-700 p-3 rounded-lg flex flex-col items-center">
            <span class="text-gray-400 text-sm">Most Distance Traveled</span>
            <span class="text-lg font-semibold text-white">{{ getMostDistanceTraveled(analytics.distance_traveled) }}</span>
          </div>
          <div class="bg-gray-700 p-3 rounded-lg flex flex-col items-center">
            <span class="text-gray-400 text-sm">Employee with Most Distance Traveled</span>
            <span class="text-lg font-semibold text-white">{{ getEmployeeWithMostDistanceTraveled(analytics.distance_traveled) }}</span>
          </div>
        
         
        </div>
        <div class="w-full h-[500px] gap-2">
          <img :src="heatmap" class="w-full h-[500px] object-cover rounded-lg border-2 border-white mt-6">
        </div>

      </div>
    </template>
  </main>
</template>

<style scoped>
.home-container {
  display: flex;
  gap: 20px;
  padding: 20px;
  height: calc(100vh - 40px);
}

.video-section {
  flex: 2;
  background-color: #1a1a1a;
  border-radius: 8px;
  overflow: hidden;
}

.video-player {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.stats-section {
  flex: 1;
  background-color: #2a2a2a;
  border-radius: 8px;
  padding: 20px;
  color: white;
}

.stats-section h2 {
  margin-bottom: 20px;
  color: #fff;
  font-size: 1.5rem;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 15px;
}

.stat-item {
  background-color: #3a3a3a;
  padding: 15px;
  border-radius: 6px;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.stat-label {
  font-size: 0.9rem;
  color: #aaa;
  margin-bottom: 5px;
}

.stat-value {
  font-size: 1.2rem;
  font-weight: bold;
  color: #fff;
}

@media (max-width: 768px) {
  .home-container {
    flex-direction: column;
  }
  
  .video-section,
  .stats-section {
    flex: none;
    height: 50vh;
  }
}
</style>
