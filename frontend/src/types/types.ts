export interface ApiResponse<T> {
    data: T;
    status: number;
    message?: string;
}
  
export interface VideoStats {
    viewers: number;
    likes: number;
    comments: number;
    duration: string;
    quality: string;
}

export interface VideoAnalysis {
    id: string;
    timestamp: string;
    duration: number;
    fps: number;
    resolution: {
        width: number;
        height: number;
    };
    format: string;
    size: number; // in bytes
}

export interface VideoAnalytics {
    viewCount: number;
    averageWatchTime: number;
    engagementRate: number;
    peakViewerCount: number;
    totalWatchTime: number;
    demographics: {
        age: Record<string, number>;
        location: Record<string, number>;
    };
}

export interface HeatmapData {
    timestamp: number;
    intensity: number;
    viewerCount: number;
    events: {
        type: string;
        count: number;
    }[];
}

export interface VideoStatus {
    status: 'processing' | 'ready' | 'failed' | 'analyzing';
    progress: number;
    error?: string;
    lastUpdated: string;
}
