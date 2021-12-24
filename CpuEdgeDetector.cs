using System;
using System.Windows.Media.Imaging;

namespace HackathonBEST
{
    public class CpuEdgeDetector : EdgeDetector
    {
        public override void Execute()
        {
            BitmapImage bitmap = new BitmapImage();  
            bitmap.BeginInit();  
            bitmap.UriSource = new Uri(FilePath);  
            bitmap.EndInit();
            OnDetectionCompleted.Invoke(bitmap);
        }
    }
}