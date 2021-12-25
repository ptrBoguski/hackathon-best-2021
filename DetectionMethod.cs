namespace HackathonBEST
{
    public enum DetectionMethod
    {
        CPU,
        GPU,
    }
    
    static class DetectionMethodExtensions 
    {
        public static EdgeDetector GetEdgeDetector(this DetectionMethod detectionMethod) 
        {
            switch (detectionMethod) 
            {
                case DetectionMethod.CPU:
                    return new CpuEdgeDetector();
                case DetectionMethod.GPU:
                    return new GpuEdgeDetector();
            }
            return null;
        }
    }
}