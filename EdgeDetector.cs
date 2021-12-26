﻿using System;
using System.Windows.Media.Imaging;

namespace HackathonBEST
{
    public abstract class EdgeDetector
    {
        public Action<BitmapImage, TimeSpan> OnDetectionCompleted;
        public abstract void Execute(string filePath, double threshold);
    }
}