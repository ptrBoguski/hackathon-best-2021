using System;
using System.Windows.Media.Imaging;

namespace HackathonBEST
{
    public abstract class EdgeDetector
    {
        public string FilePath { get; set; }
        public Action<BitmapImage> OnDetectionCompleted;
        public abstract void Execute();
    }
}