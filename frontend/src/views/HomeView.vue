<script setup lang="ts">
import { ref, onMounted } from 'vue';
import videoApi from '../api/api';
import type { VideoStats, VideoAnalytics, HeatmapData } from '../types/types';

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
const videoElement = ref<HTMLVideoElement | null>(null);
const videoId = ref<string>('your-video-id'); // Replace with actual video ID

onMounted(async () => {
  if (videoElement.value) {
    videoElement.value.autoplay = true;
    videoElement.value.muted = true;
  }
  
  // Initial data fetch
  await Promise.all([
    updateAnalytics(),
    updateHeatmap()
  ]);

  // Set up periodic updates
  // setInterval(updateAnalytics, 30000); // Update analytics every 30 seconds
  // setInterval(updateHeatmap, 10000); // Update heatmap every 10 seconds
});

const updateAnalytics = async () => {
  try {
    const response = await videoApi.getAnalytics(videoId.value);
    analytics.value = response.data;
  } catch (error) {
    console.error('Failed to fetch analytics:', error);
  }
};

const updateHeatmap = async () => {
  try {
    const response = await videoApi.getHeatmap(videoId.value);
    heatmapData.value = response.data;
  } catch (error) {
    console.error('Failed to fetch heatmap:', error);
  }
};
</script>

<template>
  <main class="w-screen h-screen flex flex-col lg:flex-row">
    <!-- Video Section -->
    <div class="w-full lg:w-2/3 bg-gray-900">
      <video
        ref="videoElement"
        class="w-full h-full object-cover"
        controls
        playsinline
      >
        <source :src="videoUrl" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </div>
    
    <!-- Stats Section -->
    <div class="w-full lg:w-1/3 p-4 overflow-y-auto">
      <h2 class="text-xl font-bold mb-4 text-white">Stats</h2>
      <div class="grid grid-cols-2 gap-3">
        <div class="bg-gray-700 p-3 rounded-lg flex flex-col items-center">
          <span class="text-gray-400 text-sm">Viewers</span>
          <span class="text-lg font-semibold text-white">{{ videoStats.viewers }}</span>
        </div>
        <div class="bg-gray-700 p-3 rounded-lg flex flex-col items-center">
          <span class="text-gray-400 text-sm">Likes</span>
          <span class="text-lg font-semibold text-white">{{ videoStats.likes }}</span>
        </div>
        <div class="bg-gray-700 p-3 rounded-lg flex flex-col items-center">
          <span class="text-gray-400 text-sm">Comments</span>
          <span class="text-lg font-semibold text-white">{{ videoStats.comments }}</span>
        </div>
        <div class="bg-gray-700 p-3 rounded-lg flex flex-col items-center">
          <span class="text-gray-400 text-sm">Duration</span>
          <span class="text-lg font-semibold text-white">{{ videoStats.duration }}</span>
        </div>
        <div class="bg-gray-700 p-3 rounded-lg flex flex-col items-center">
          <span class="text-gray-400 text-sm">Quality</span>
          <span class="text-lg font-semibold text-white">{{ videoStats.quality }}</span>
        </div>
      </div>

      <!-- Analytics Section -->
      <template v-if="analytics">
        <h2 class="text-xl font-bold my-4 text-white">Analytics</h2>
        <div class="grid grid-cols-2 gap-3">
          <div class="bg-gray-700 p-3 rounded-lg flex flex-col items-center">
            <span class="text-gray-400 text-sm">Total Views</span>
            <span class="text-lg font-semibold text-white">{{ analytics.viewCount }}</span>
          </div>
          <div class="bg-gray-700 p-3 rounded-lg flex flex-col items-center">
            <span class="text-gray-400 text-sm">Avg Watch Time</span>
            <span class="text-lg font-semibold text-white">{{ analytics.averageWatchTime }}s</span>
          </div>
          <div class="bg-gray-700 p-3 rounded-lg flex flex-col items-center">
            <span class="text-gray-400 text-sm">Engagement Rate</span>
            <span class="text-lg font-semibold text-white">{{ (analytics.engagementRate * 100).toFixed(1) }}%</span>
          </div>
          <div class="bg-gray-700 p-3 rounded-lg flex flex-col items-center">
            <span class="text-gray-400 text-sm">Peak Viewers</span>
            <span class="text-lg font-semibold text-white">{{ analytics.peakViewerCount }}</span>
          </div>
        </div>
      </template>
    </div>
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
