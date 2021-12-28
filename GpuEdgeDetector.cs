using System;
using System.Windows.Media.Imaging;
using System.Runtime.InteropServices;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;

namespace HackathonBEST
{
    
    
    public class GpuEdgeDetector: EdgeDetector
    {
        [DllImport(@"NvidiaKernel.dll",  CallingConvention=CallingConvention.Cdecl)]
        private static extern void run(byte[] red, byte[] x, int width, int height, float[] ms);
        public override void Execute(string filePath, float[] xMask, float[] yMask, double threshold, double sigma, bool gaussian, bool supression)
        {
            var start = DateTime.Now;
            Bitmap i = new Bitmap(filePath);
            Console.WriteLine(i);
            int width = i.Width;
            int height = i.Height;
            Rectangle rect = new Rectangle(0,0, width, height);
            BitmapData bmpData = i.LockBits(rect, 
            ImageLockMode.ReadWrite, PixelFormat.Format32bppRgb);
            IntPtr ptr = bmpData.Scan0;
            int bytes  = i.Width * i.Height * 4;
            byte[] rgbValues = new byte[bytes];
            Marshal.Copy(ptr, rgbValues, 0, bytes);
            Console.WriteLine("skonczone kopiowanie");
            Console.WriteLine("odpalanie kernela");
            byte [] greyscale = new byte[bytes];
            float [] ms = new float[1];
            run(rgbValues, greyscale, width,height, ms);
            Console.WriteLine("zakonczenie kernela");
            Console.WriteLine("Kernel took: " + ms[0] +" ms");
            Bitmap result = new Bitmap(width, height);
            BitmapData resultData = result.LockBits(rect, 
            ImageLockMode.ReadWrite, PixelFormat.Format32bppRgb);
            Marshal.Copy(greyscale, 0, resultData.Scan0, bytes);
            var stop = DateTime.Now;
            result.UnlockBits(resultData);
            i.UnlockBits(bmpData);
            TimeSpan diff = stop - start;
            Console.Write(diff);
            using (MemoryStream mem = new MemoryStream())
            {
                result.Save(mem, ImageFormat.Png);
                mem.Position = 0;
                BitmapImage bitmapImage = new BitmapImage();
                bitmapImage.BeginInit();
                bitmapImage.StreamSource = mem;
                bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
                bitmapImage.EndInit();
                OnDetectionCompleted.Invoke(bitmapImage, diff);
            }
        }
    }
}
