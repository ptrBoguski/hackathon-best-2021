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
    [DllImport(@"C:\Users\Shadow\asdasdasd\hackathon-best-2021\NvidiaKernel.dll",  CallingConvention=CallingConvention.Cdecl)]
    private static extern void run(byte[] red, byte[] x, int width, int height);
        public override void Execute()
        {
            BitmapImage bitmap = new BitmapImage();  
            bitmap.BeginInit();  
            bitmap.UriSource = new Uri(FilePath);  
            bitmap.EndInit();
            OnDetectionCompleted.Invoke(bitmap);
            
            var start = DateTime.Now;
            // Bitmap i = new Bitmap("51766629975_1eec90d220_o.jpg");
            Bitmap i = new Bitmap(FilePath);
            Console.WriteLine(i);
            int width = i.Width;
            int height = i.Height;
            Rectangle rect = new Rectangle(0,0, width, height);
            System.Drawing.Imaging.BitmapData bmpData = i.LockBits(rect, 
                System.Drawing.Imaging.ImageLockMode.ReadWrite, PixelFormat.Format32bppRgb);
            IntPtr ptr = bmpData.Scan0;
            int bytes  = i.Width * i.Height * 4;
            byte[] rgbValues = new byte[bytes];
            System.Runtime.InteropServices.Marshal.Copy(ptr, rgbValues, 0, bytes);
            Console.WriteLine("skonczone kopiowanie");
            Console.WriteLine("odpalanie kernela");
            byte [] greyscale = new byte[bytes];
            run(rgbValues, greyscale, width,height);
            Console.WriteLine("zakonczenie kernela");
            Bitmap result = new Bitmap(width, height);
            BitmapData resultData = result.LockBits(rect, 
            ImageLockMode.ReadWrite, PixelFormat.Format32bppRgb);
            Marshal.Copy(greyscale, 0, resultData.Scan0, bytes);
            var stop = DateTime.Now;
            result.UnlockBits(resultData);
            i.UnlockBits(bmpData);
            result.Save("result.png",ImageFormat.Png);
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
                OnDetectionCompleted.Invoke(bitmapImage);
            }
        }
    }
}
