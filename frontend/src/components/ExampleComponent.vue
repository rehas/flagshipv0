<template>
  <div>
    <h2>Example Component</h2>
    <button @click="fetchData">Fetch Data</button>
    <div v-if="loading">Loading...</div>
    <div v-else-if="error">{{ error }}</div>
    <div v-else>
      <pre>{{ data }}</pre>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import apiClient from '../utils/axios';
import { ApiResponse } from '../types/types';

interface ExampleData {
  id: number;
  name: string;
  // Add other properties based on your API response
}

const data = ref<ExampleData | null>(null);
const loading = ref<boolean>(false);
const error = ref<string | null>(null);

const fetchData = async (): Promise<void> => {
  loading.value = true;
  error.value = null;
  
  try {
    const response = await apiClient.get<ApiResponse<ExampleData>>('/api/endpoint');
    data.value = response.data.data;
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'An error occurred';
  } finally {
    loading.value = false;
  }
};
</script>

<style scoped>
div {
  padding: 20px;
}
button {
  padding: 10px 20px;
  background-color: #4CAF50;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}
button:hover {
  background-color: #45a049;
}
</style> 