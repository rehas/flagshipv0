import apiClient from '../utils/axios';
import type { ApiResponse, Video, VideoStats, VideoAnalysis, VideoAnalytics, VideoStatus, HeatmapData, VideoListItem } from '../types/types';


// API Functions
export const videoApi = {
    getVideo: async (videoId: string): Promise<ApiResponse<Video>> => {
        const response = await apiClient.get<ApiResponse<Video>>(`/video/${videoId}`);
        return response.data;
    },
  // List all available videos
  listVideos: async (): Promise<VideoListItem[]> => {
    const response = await apiClient.get<VideoListItem[]>('/videos');
    return response.data;
  },

  // Analyze video endpoint
  analyzeVideo: async (videoId: string): Promise<ApiResponse<VideoAnalysis>> => {
    const response = await apiClient.post<ApiResponse<VideoAnalysis>>('/analytics', { video_name: videoId });
    return response.data;
  },

  // Get video analytics
  getAnalytics: async (videoId: string, timeRange?: { start: string; end: string }): Promise<VideoAnalytics> => {
    const response = await apiClient.get<VideoAnalytics>('/analytics', {
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