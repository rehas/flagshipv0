import apiClient from '../utils/axios';
import type { ApiResponse, VideoStats, VideoAnalysis, VideoAnalytics, VideoStatus, HeatmapData } from '../types/types';


// API Functions
export const videoApi = {
  // Analyze video endpoint
  analyzeVideo: async (videoId: string): Promise<ApiResponse<VideoAnalysis>> => {
    const response = await apiClient.post<ApiResponse<VideoAnalysis>>('/analyze-video', {
      videoId
    });
    return response.data;
  },

  // Get video analytics
  getAnalytics: async (videoId: string, timeRange?: { start: string; end: string }): Promise<ApiResponse<VideoAnalytics>> => {
    const response = await apiClient.get<ApiResponse<VideoAnalytics>>('/analytics', {
      params: {
        videoId,
        ...timeRange
      }
    });
    return response.data;
  },

  // Get video heatmap data
  getHeatmap: async (videoId: string): Promise<ApiResponse<HeatmapData[]>> => {
    const response = await apiClient.get<ApiResponse<HeatmapData[]>>('/heatmap', {
      params: { videoId }
    });
    return response.data;
  },

  // Get video processing status
  getStatus: async (videoId: string): Promise<ApiResponse<VideoStatus>> => {
    const response = await apiClient.get<ApiResponse<VideoStatus>>('/status', {
      params: { videoId }
    });
    return response.data;
  },

  // Get video stats (current viewers, likes, etc.)
  getStats: async (videoId: string): Promise<ApiResponse<VideoStats>> => {
    const response = await apiClient.get<ApiResponse<VideoStats>>(`/video/${videoId}/stats`);
    return response.data;
  },

  // Update video stats (e.g., increment likes)
  updateStats: async (videoId: string, data: Partial<VideoStats>): Promise<ApiResponse<VideoStats>> => {
    const response = await apiClient.patch<ApiResponse<VideoStats>>(`/video/${videoId}/stats`, data);
    return response.data;
  }
};

export default videoApi; 